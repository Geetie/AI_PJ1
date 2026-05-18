import os
import sys
import gc
import time
import json
import copy
import traceback
import threading
from datetime import datetime
from collections import OrderedDict

import torch as t
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config, GPU_PLATFORM, TOTAL_VRAM_GB, COMPILE_AVAILABLE
from models import create_model
from models.multihead import DigitsResnet101
from models.ctc import CTCModel
from models.transformer import TransformerDigitsModel
from utils.compile_utils import (
    is_compile_available, try_compile_model, warmup_model,
    get_raw_model, measure_compile_performance, configure_dynamo_cache,
    CompileLogger, configure_compile_cache, selective_compile,
    WARMUP_STRATEGIES, run_compile_audit
)

ALL_MODEL_TYPES = ['fpn_multihead', 'ctc', 'transformer']
ALL_COMPILE_MODES = ['default', 'reduce-overhead', 'max-autotune']
ALL_DTYPES = [t.float32, t.float16]
if hasattr(t, 'bfloat16'):
    ALL_DTYPES.append(t.bfloat16)

INPUT_SIZES = [
    (224, 224),
    (288, 288),
    (320, 320),
    (352, 352),
    (384, 384),
    (416, 416),
]

BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128]

DYNAMIC_OPTIONS = [True, False]
FULLGRAPH_OPTIONS = [True, False]

NUM_WARMUP_ITERS = 3
NUM_BENCH_ITERS = 20
CORRECTNESS_ATOL = 1e-2
CORRECTNESS_RTOL = 1e-2


class AuditRecorder:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results = OrderedDict()
        self._lock = threading.Lock()

    def record(self, category, test_name, data):
        with self._lock:
            if category not in self.results:
                self.results[category] = OrderedDict()
            self.results[category][test_name] = data

    def save(self):
        path = os.path.join(self.output_dir, f'compile_audit_{self.timestamp}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f'\n[SAVE] Full audit results saved to {path}')
        return path

    def print_summary(self):
        print('\n' + '=' * 80)
        print('COMPILE AUDIT SUMMARY')
        print('=' * 80)
        for category, tests in self.results.items():
            print(f'\n--- {category} ---')
            for test_name, data in tests.items():
                status = data.get('status', 'unknown')
                if status == 'pass':
                    key_metrics = []
                    if 'compile_time_s' in data:
                        key_metrics.append(f"compile={data['compile_time_s']:.2f}s")
                    if 'warmup_time_s' in data:
                        key_metrics.append(f"warmup={data['warmup_time_s']:.2f}s")
                    if 'total_setup_time_s' in data:
                        key_metrics.append(f"setup={data['total_setup_time_s']:.2f}s")
                    if 'avg_latency_ms' in data:
                        key_metrics.append(f"avg={data['avg_latency_ms']:.2f}ms")
                    if 'first_latency_s' in data:
                        key_metrics.append(f"first={data['first_latency_s']:.3f}s")
                    if 'gpu_mem_peak_gb' in data:
                        key_metrics.append(f"peak_mem={data['gpu_mem_peak_gb']:.2f}GB")
                    if 'max_diff' in data:
                        key_metrics.append(f"max_diff={data['max_diff']:.6f}")
                    print(f'  [PASS] {test_name}: {", ".join(key_metrics)}')
                elif status == 'fail':
                    err = data.get('error', 'unknown')[:80]
                    print(f'  [FAIL] {test_name}: {err}')
                elif status == 'skip':
                    reason = data.get('reason', '')
                    print(f'  [SKIP] {test_name}: {reason}')
                else:
                    print(f'  [{status.upper()}] {test_name}')


def _gpu_mem_gb():
    if t.cuda.is_available():
        return t.cuda.memory_allocated() / (1024**3), t.cuda.max_memory_allocated() / (1024**3)
    return 0, 0


def _reset_gpu_stats():
    gc.collect()
    if t.cuda.is_available():
        t.cuda.empty_cache()
        t.cuda.reset_peak_memory_stats()


def _create_model(model_type, device):
    model = create_model(model_type).to(device)
    model.eval()
    return model


def _make_dummy(input_shape, device, dtype=t.float32):
    bs, h, w = input_shape
    return t.randn(bs, 3, h, w, device=device, dtype=dtype)


def _eager_reference_output(model, device, input_shape, dtype=t.float32):
    raw = get_raw_model(model)
    raw.eval()
    dummy = _make_dummy(input_shape, device, dtype)
    with t.no_grad():
        out = raw(dummy)
    del dummy
    return out


def _check_correctness(compiled_out, eager_out, label=''):
    if isinstance(eager_out, (tuple, list)):
        max_diff = 0.0
        for i, (co, eo) in enumerate(zip(compiled_out, eager_out)):
            if isinstance(co, t.Tensor) and isinstance(eo, t.Tensor):
                co_cpu = co.detach().cpu().float()
                eo_cpu = eo.detach().cpu().float()
                if co_cpu.shape != eo_cpu.shape:
                    return False, float('inf'), f'{label} shape mismatch at idx {i}'
                diff = (co_cpu - eo_cpu).abs().max().item()
                max_diff = max(max_diff, diff)
        return max_diff < CORRECTNESS_ATOL, max_diff, label
    elif isinstance(eager_out, t.Tensor):
        co_cpu = compiled_out.detach().cpu().float()
        eo_cpu = eager_out.detach().cpu().float()
        if co_cpu.shape != eo_cpu.shape:
            return False, float('inf'), f'{label} shape mismatch'
        max_diff = (co_cpu - eo_cpu).abs().max().item()
        close = t.allclose(co_cpu, eo_cpu, atol=CORRECTNESS_ATOL, rtol=CORRECTNESS_RTOL)
        return close, max_diff, label
    return True, 0.0, label


def _measure_perf(model, device, input_shape, dtype=t.float32,
                  num_warmup=NUM_WARMUP_ITERS, num_iters=NUM_BENCH_ITERS):
    from torch.amp import autocast
    dev_type = device.type if isinstance(device, t.device) else device
    use_amp = (dev_type == 'cuda') and (dtype == t.float32)

    dummy = _make_dummy(input_shape, device, dtype)

    t0 = time.time()
    with t.no_grad():
        if use_amp:
            with autocast(dev_type):
                _ = model(dummy)
        else:
            _ = model(dummy)
    if t.cuda.is_available():
        t.cuda.synchronize()
    first_latency = time.time() - t0

    for _ in range(num_warmup):
        with t.no_grad():
            if use_amp:
                with autocast(dev_type):
                    _ = model(dummy)
            else:
                _ = model(dummy)
        if t.cuda.is_available():
            t.cuda.synchronize()

    latencies = []
    for _ in range(num_iters):
        t0 = time.time()
        with t.no_grad():
            if use_amp:
                with autocast(dev_type):
                    _ = model(dummy)
            else:
                _ = model(dummy)
        if t.cuda.is_available():
            t.cuda.synchronize()
        latencies.append(time.time() - t0)

    latencies_ms = np.array(latencies) * 1000
    alloc_gb, peak_gb = _gpu_mem_gb()

    del dummy
    if t.cuda.is_available():
        t.cuda.empty_cache()

    return {
        'first_latency_s': first_latency,
        'avg_latency_ms': float(latencies_ms.mean()),
        'p50_latency_ms': float(np.percentile(latencies_ms, 50)),
        'p99_latency_ms': float(np.percentile(latencies_ms, 99)),
        'min_latency_ms': float(latencies_ms.min()),
        'max_latency_ms': float(latencies_ms.max()),
        'gpu_mem_allocated_gb': alloc_gb,
        'gpu_mem_peak_gb': peak_gb,
    }


def _compile_and_measure(model, device, input_shape, mode='default',
                         dynamic=True, fullgraph=False, dtype=t.float32,
                         check_correct=True, forward_fn=None):
    raw_model = get_raw_model(model)
    eager_out = None
    if check_correct:
        try:
            eager_out = _eager_reference_output(raw_model, device, input_shape, dtype)
        except Exception:
            check_correct = False

    compile_start = time.time()
    compiled_model, ok = try_compile_model(model, mode=mode, dynamic=dynamic, fullgraph=fullgraph)
    compile_time = time.time() - compile_start

    if not ok:
        return {
            'status': 'fail',
            'error': 'torch.compile() returned failure',
            'compile_time_s': compile_time,
        }

    warmup_start = time.time()
    try:
        warmup_model(compiled_model, device, [input_shape],
                     forward_fn=forward_fn, num_steps=2, use_amp=(dtype == t.float32))
    except Exception as e:
        err_str = str(e).lower()
        if 'compile' in err_str or 'triton' in err_str or 'inductor' in err_str:
            raw = get_raw_model(compiled_model)
            return {
                'status': 'fail',
                'error': f'warmup compile error: {e}',
                'compile_time_s': compile_time,
            }
        warmup_time = time.time() - warmup_start
        return {
            'status': 'fail',
            'error': f'warmup error: {e}',
            'compile_time_s': compile_time,
            'warmup_time_s': warmup_time,
        }
    warmup_time = time.time() - warmup_start

    perf = _measure_perf(compiled_model, device, input_shape, dtype=dtype)

    correct = None
    max_diff = None
    if check_correct and eager_out is not None:
        try:
            dummy = _make_dummy(input_shape, device, dtype)
            with t.no_grad():
                compiled_out = compiled_model(dummy)
            del dummy
            correct, max_diff, _ = _check_correctness(compiled_out, eager_out)
        except Exception as e:
            correct = False
            max_diff = float('inf')

    return {
        'status': 'pass' if correct is None or correct else 'incorrect',
        'compile_time_s': compile_time,
        'warmup_time_s': warmup_time,
        'total_setup_time_s': compile_time + warmup_time,
        'correct': correct,
        'max_diff': max_diff,
        **perf,
    }


def audit_compile_modes(recorder, device):
    print('\n' + '=' * 80)
    print('AUDIT 1: Compile Modes (default / reduce-overhead / max-autotune)')
    print('=' * 80)

    for model_type in ALL_MODEL_TYPES:
        for mode in ALL_COMPILE_MODES:
            test_name = f'{model_type}_{mode}'
            print(f'  Testing {test_name}...', end=' ', flush=True)
            _reset_gpu_stats()

            try:
                model = _create_model(model_type, device)
                input_shape = (min(config.batch_size, 16), config.input_height, config.input_width)
                result = _compile_and_measure(model, device, input_shape, mode=mode)
                recorder.record('compile_modes', test_name, result)
                status = result['status']
                extra = ''
                if status == 'pass':
                    extra = f"setup={result['total_setup_time_s']:.1f}s avg={result['avg_latency_ms']:.2f}ms"
                print(f'{status} {extra}')
            except Exception as e:
                recorder.record('compile_modes', test_name, {'status': 'fail', 'error': str(e)})
                print(f'fail: {e}')
            finally:
                try:
                    del model
                except Exception:
                    pass
                _reset_gpu_stats()


def audit_input_dtypes(recorder, device):
    print('\n' + '=' * 80)
    print('AUDIT 2: Input Data Types (float32 / float16 / bfloat16)')
    print('=' * 80)

    model_type = 'fpn_multihead'
    mode = 'default'

    for dtype in ALL_DTYPES:
        dtype_name = str(dtype).split('.')[-1]
        test_name = f'{model_type}_{dtype_name}'
        print(f'  Testing {test_name}...', end=' ', flush=True)
        _reset_gpu_stats()

        try:
            model = _create_model(model_type, device)
            if dtype != t.float32:
                model = model.to(dtype)
            input_shape = (min(config.batch_size, 16), config.input_height, config.input_width)
            result = _compile_and_measure(model, device, input_shape, mode=mode, dtype=dtype)
            recorder.record('input_dtypes', test_name, result)
            status = result['status']
            extra = ''
            if status == 'pass':
                extra = f"setup={result['total_setup_time_s']:.1f}s avg={result['avg_latency_ms']:.2f}ms"
            print(f'{status} {extra}')
        except Exception as e:
            recorder.record('input_dtypes', test_name, {'status': 'fail', 'error': str(e)})
            print(f'fail: {e}')
        finally:
            try:
                del model
            except Exception:
                pass
            _reset_gpu_stats()


def audit_input_sizes(recorder, device):
    print('\n' + '=' * 80)
    print('AUDIT 3: Input Sizes (224x224 to 416x416)')
    print('=' * 80)

    model_type = 'fpn_multihead'
    mode = 'default'
    bs = min(config.batch_size, 16)

    model = _create_model(model_type, device)
    compile_start = time.time()
    model, ok = try_compile_model(model, mode=mode, dynamic=True)
    compile_time = time.time() - compile_start

    if not ok:
        recorder.record('input_sizes', 'compile_failed',
                        {'status': 'fail', 'error': 'compile failed'})
        del model
        return

    first_shape = (bs, INPUT_SIZES[0][0], INPUT_SIZES[0][1])
    warmup_start = time.time()
    warmup_model(model, device, [first_shape], num_steps=2)
    warmup_time = time.time() - warmup_start

    for h, w in INPUT_SIZES:
        input_shape = (bs, h, w)
        test_name = f'bs{bs}_{h}x{w}'
        print(f'  Testing {test_name}...', end=' ', flush=True)

        try:
            perf = _measure_perf(model, device, input_shape)
            perf['compile_time_s'] = compile_time
            perf['warmup_time_s'] = warmup_time
            perf['total_setup_time_s'] = compile_time + warmup_time
            perf['status'] = 'pass'
            recorder.record('input_sizes', test_name, perf)
            print(f"pass avg={perf['avg_latency_ms']:.2f}ms first={perf['first_latency_s']:.3f}s")
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                recorder.record('input_sizes', test_name, {'status': 'oom', 'error': str(e)})
                print('OOM')
                t.cuda.empty_cache()
            else:
                recorder.record('input_sizes', test_name, {'status': 'fail', 'error': str(e)})
                print(f'fail: {e}')
        except Exception as e:
            recorder.record('input_sizes', test_name, {'status': 'fail', 'error': str(e)})
            print(f'fail: {e}')

    del model
    _reset_gpu_stats()


def audit_batch_sizes(recorder, device):
    print('\n' + '=' * 80)
    print('AUDIT 4: Batch Sizes (1 to 128)')
    print('=' * 80)

    model_type = 'fpn_multihead'
    mode = 'default'
    h, w = config.input_height, config.input_width

    model = _create_model(model_type, device)
    compile_start = time.time()
    model, ok = try_compile_model(model, mode=mode, dynamic=True)
    compile_time = time.time() - compile_start

    if not ok:
        recorder.record('batch_sizes', 'compile_failed',
                        {'status': 'fail', 'error': 'compile failed'})
        del model
        return

    first_shape = (min(BATCH_SIZES), h, w)
    warmup_start = time.time()
    warmup_model(model, device, [first_shape], num_steps=2)
    warmup_time = time.time() - warmup_start

    for bs in BATCH_SIZES:
        input_shape = (bs, h, w)
        test_name = f'bs{bs}_{h}x{w}'
        print(f'  Testing {test_name}...', end=' ', flush=True)

        try:
            perf = _measure_perf(model, device, input_shape)
            perf['compile_time_s'] = compile_time
            perf['warmup_time_s'] = warmup_time
            perf['total_setup_time_s'] = compile_time + warmup_time
            perf['status'] = 'pass'
            recorder.record('batch_sizes', test_name, perf)
            print(f"pass avg={perf['avg_latency_ms']:.2f}ms first={perf['first_latency_s']:.3f}s")
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                recorder.record('batch_sizes', test_name, {'status': 'oom', 'error': str(e)})
                print('OOM')
                t.cuda.empty_cache()
            else:
                recorder.record('batch_sizes', test_name, {'status': 'fail', 'error': str(e)})
                print(f'fail: {e}')
        except Exception as e:
            recorder.record('batch_sizes', test_name, {'status': 'fail', 'error': str(e)})
            print(f'fail: {e}')

    del model
    _reset_gpu_stats()


def audit_dynamic_vs_static(recorder, device):
    print('\n' + '=' * 80)
    print('AUDIT 5: Dynamic vs Static Shapes')
    print('=' * 80)

    model_type = 'fpn_multihead'
    mode = 'default'

    for dynamic in DYNAMIC_OPTIONS:
        for fullgraph in FULLGRAPH_OPTIONS:
            test_name = f'dynamic={dynamic}_fullgraph={fullgraph}'
            print(f'  Testing {test_name}...', end=' ', flush=True)
            _reset_gpu_stats()

            try:
                model = _create_model(model_type, device)
                input_shape = (min(config.batch_size, 16), config.input_height, config.input_width)
                result = _compile_and_measure(model, device, input_shape, mode=mode,
                                              dynamic=dynamic, fullgraph=fullgraph)
                recorder.record('dynamic_vs_static', test_name, result)
                status = result['status']
                extra = ''
                if status == 'pass':
                    extra = f"setup={result['total_setup_time_s']:.1f}s avg={result['avg_latency_ms']:.2f}ms"
                print(f'{status} {extra}')
            except Exception as e:
                recorder.record('dynamic_vs_static', test_name, {'status': 'fail', 'error': str(e)})
                print(f'fail: {e}')
            finally:
                try:
                    del model
                except Exception:
                    pass
                _reset_gpu_stats()


def audit_model_types(recorder, device):
    print('\n' + '=' * 80)
    print('AUDIT 6: All Model Types with Compile')
    print('=' * 80)

    mode = 'default'

    for model_type in ALL_MODEL_TYPES:
        test_name = f'{model_type}_default'
        print(f'  Testing {test_name}...', end=' ', flush=True)
        _reset_gpu_stats()

        try:
            model = _create_model(model_type, device)
            input_shape = (min(config.batch_size, 16), config.input_height, config.input_width)
            result = _compile_and_measure(model, device, input_shape, mode=mode)
            recorder.record('model_types', test_name, result)
            status = result['status']
            extra = ''
            if status == 'pass':
                extra = f"setup={result['total_setup_time_s']:.1f}s avg={result['avg_latency_ms']:.2f}ms correct={result.get('correct')}"
            print(f'{status} {extra}')
        except Exception as e:
            recorder.record('model_types', test_name, {'status': 'fail', 'error': str(e)})
            print(f'fail: {e}')
        finally:
            try:
                del model
            except Exception:
                pass
            _reset_gpu_stats()


def audit_correctness(recorder, device):
    print('\n' + '=' * 80)
    print('AUDIT 7: Correctness Verification (Compiled vs Eager)')
    print('=' * 80)

    for model_type in ALL_MODEL_TYPES:
        for mode in ['default', 'reduce-overhead']:
            test_name = f'{model_type}_{mode}'
            print(f'  Testing {test_name}...', end=' ', flush=True)
            _reset_gpu_stats()

            try:
                model = _create_model(model_type, device)
                input_shape = (4, config.input_height, config.input_width)

                eager_out = _eager_reference_output(model, device, input_shape)

                compiled_model, ok = try_compile_model(model, mode=mode, dynamic=True)
                if not ok:
                    recorder.record('correctness', test_name,
                                    {'status': 'fail', 'error': 'compile failed'})
                    print('compile failed')
                    del model
                    continue

                warmup_model(compiled_model, device, [input_shape], num_steps=2)

                dummy = _make_dummy(input_shape, device)
                with t.no_grad():
                    compiled_out = compiled_model(dummy)
                del dummy

                correct, max_diff, detail = _check_correctness(compiled_out, eager_out, test_name)
                result = {
                    'status': 'pass' if correct else 'incorrect',
                    'correct': correct,
                    'max_diff': max_diff,
                    'detail': detail,
                }
                recorder.record('correctness', test_name, result)
                print(f"{'pass' if correct else 'INCORRECT'} max_diff={max_diff:.6f}")
            except Exception as e:
                recorder.record('correctness', test_name, {'status': 'fail', 'error': str(e)})
                print(f'fail: {e}')
            finally:
                try:
                    del model
                except Exception:
                    pass
                _reset_gpu_stats()


def audit_warmup_strategies(recorder, device):
    print('\n' + '=' * 80)
    print('AUDIT 8: Warmup Strategies Comparison')
    print('=' * 80)

    model_type = 'fpn_multihead'
    input_shape = (min(config.batch_size, 16), config.input_height, config.input_width)

    strategy_configs = [
        {
            'name': 'no_warmup',
            'fn': lambda model, dev, shape: _strategy_no_warmup(model, dev, shape),
        },
        {
            'name': 'single_shape_2step',
            'fn': lambda model, dev, shape: _strategy_single_shape(model, dev, shape, steps=2),
        },
        {
            'name': 'single_shape_5step',
            'fn': lambda model, dev, shape: _strategy_single_shape(model, dev, shape, steps=5),
        },
        {
            'name': 'multi_shape_tta',
            'fn': lambda model, dev, shape: _strategy_multi_tta(model, dev, shape),
        },
        {
            'name': 'fixed_shape_no_dynamic',
            'fn': lambda model, dev, shape: _strategy_fixed_shape(model, dev, shape),
        },
        {
            'name': 'cache_optimized',
            'fn': lambda model, dev, shape: _strategy_cache_optimized(model, dev, shape),
        },
        {
            'name': 'selective_backbone',
            'fn': lambda model, dev, shape: _strategy_selective(model, dev, shape, ['backbone']),
        },
        {
            'name': 'max_autotune',
            'fn': lambda model, dev, shape: _strategy_max_autotune(model, dev, shape),
        },
        {
            'name': 'parallel_submodule_compile',
            'fn': lambda model, dev, shape: _strategy_parallel_submodule(model, dev, shape),
        },
        {
            'name': 'shape_hint_preregister',
            'fn': lambda model, dev, shape: _strategy_shape_hint(model, dev, shape),
        },
        {
            'name': 'mixed_eager_compile',
            'fn': lambda model, dev, shape: _strategy_mixed_eager_compile(model, dev, shape),
        },
        {
            'name': 'reduce_overhead_mode',
            'fn': lambda model, dev, shape: _strategy_reduce_overhead(model, dev, shape),
        },
        {
            'name': 'fullgraph_mode',
            'fn': lambda model, dev, shape: _strategy_fullgraph(model, dev, shape),
        },
        {
            'name': 'inductor_config_tuned',
            'fn': lambda model, dev, shape: _strategy_inductor_config_tuned(model, dev, shape),
        },
    ]

    for strat in strategy_configs:
        name = strat['name']
        print(f'  Strategy: {name}...', end=' ', flush=True)
        _reset_gpu_stats()

        try:
            model = _create_model(model_type, device)
            result = strat['fn'](model, device, input_shape)
            recorder.record('warmup_strategies', name, result)
            status = result.get('status', 'unknown')
            extra = ''
            if status == 'pass':
                extra = (f"setup={result.get('total_setup_time_s', 0):.1f}s "
                         f"avg={result.get('avg_latency_ms', 0):.2f}ms "
                         f"first={result.get('first_latency_s', 0):.3f}s")
            print(extra if status == 'pass' else status)
        except Exception as e:
            recorder.record('warmup_strategies', name, {'status': 'fail', 'error': str(e)})
            print(f'fail: {e}')
        finally:
            try:
                del model
            except Exception:
                pass
            _reset_gpu_stats()


def _strategy_no_warmup(model, device, input_shape):
    compile_start = time.time()
    model, ok = try_compile_model(model, mode='default', dynamic=True)
    compile_time = time.time() - compile_start
    if not ok:
        return {'status': 'fail', 'error': 'compile failed'}

    perf = _measure_perf(model, device, input_shape, num_warmup=0, num_iters=NUM_BENCH_ITERS)
    return {
        'status': 'pass',
        'compile_time_s': compile_time,
        'warmup_time_s': 0,
        'total_setup_time_s': compile_time,
        **perf,
    }


def _strategy_single_shape(model, device, input_shape, steps=2):
    compile_start = time.time()
    model, ok = try_compile_model(model, mode='default', dynamic=True)
    compile_time = time.time() - compile_start
    if not ok:
        return {'status': 'fail', 'error': 'compile failed'}

    warmup_start = time.time()
    warmup_model(model, device, [input_shape], num_steps=steps)
    warmup_time = time.time() - warmup_start

    perf = _measure_perf(model, device, input_shape)
    return {
        'status': 'pass',
        'compile_time_s': compile_time,
        'warmup_time_s': warmup_time,
        'total_setup_time_s': compile_time + warmup_time,
        **perf,
    }


def _strategy_multi_tta(model, device, input_shape):
    compile_start = time.time()
    model, ok = try_compile_model(model, mode='default', dynamic=True)
    compile_time = time.time() - compile_start
    if not ok:
        return {'status': 'fail', 'error': 'compile failed'}

    bs = input_shape[0]
    shapes = [(bs, s, s) for s in config.tta_sizes]
    shapes.append(input_shape)

    warmup_start = time.time()
    warmup_model(model, device, shapes, num_steps=1)
    warmup_time = time.time() - warmup_start

    perf = _measure_perf(model, device, input_shape)
    return {
        'status': 'pass',
        'compile_time_s': compile_time,
        'warmup_time_s': warmup_time,
        'total_setup_time_s': compile_time + warmup_time,
        'num_tta_shapes': len(shapes),
        **perf,
    }


def _strategy_fixed_shape(model, device, input_shape):
    compile_start = time.time()
    model, ok = try_compile_model(model, mode='default', dynamic=False)
    compile_time = time.time() - compile_start
    if not ok:
        return {'status': 'fail', 'error': 'compile failed'}

    warmup_start = time.time()
    warmup_model(model, device, [input_shape], num_steps=2)
    warmup_time = time.time() - warmup_start

    perf = _measure_perf(model, device, input_shape)
    return {
        'status': 'pass',
        'compile_time_s': compile_time,
        'warmup_time_s': warmup_time,
        'total_setup_time_s': compile_time + warmup_time,
        **perf,
    }


def _strategy_cache_optimized(model, device, input_shape):
    configure_compile_cache()
    configure_dynamo_cache(cache_size_limit=512, accumulated_cache_size_limit=1024)

    compile_start = time.time()
    model, ok = try_compile_model(model, mode='default', dynamic=True)
    compile_time = time.time() - compile_start
    if not ok:
        return {'status': 'fail', 'error': 'compile failed'}

    warmup_start = time.time()
    warmup_model(model, device, [input_shape], num_steps=2)
    warmup_time = time.time() - warmup_start

    perf = _measure_perf(model, device, input_shape)
    return {
        'status': 'pass',
        'compile_time_s': compile_time,
        'warmup_time_s': warmup_time,
        'total_setup_time_s': compile_time + warmup_time,
        **perf,
    }


def _strategy_selective(model, device, input_shape, submodules):
    compile_start = time.time()
    model, ok = selective_compile(model, mode='default', compile_submodules=submodules)
    compile_time = time.time() - compile_start
    if not ok:
        return {'status': 'fail', 'error': 'selective compile failed'}

    warmup_start = time.time()
    warmup_model(model, device, [input_shape], num_steps=2)
    warmup_time = time.time() - warmup_start

    perf = _measure_perf(model, device, input_shape)
    return {
        'status': 'pass',
        'compile_time_s': compile_time,
        'warmup_time_s': warmup_time,
        'total_setup_time_s': compile_time + warmup_time,
        'compiled_submodules': submodules,
        **perf,
    }


def _strategy_max_autotune(model, device, input_shape):
    compile_start = time.time()
    model, ok = try_compile_model(model, mode='max-autotune', dynamic=True)
    compile_time = time.time() - compile_start
    if not ok:
        return {'status': 'fail', 'error': 'compile failed'}

    warmup_start = time.time()
    warmup_model(model, device, [input_shape], num_steps=3)
    warmup_time = time.time() - warmup_start

    perf = _measure_perf(model, device, input_shape)
    return {
        'status': 'pass',
        'compile_time_s': compile_time,
        'warmup_time_s': warmup_time,
        'total_setup_time_s': compile_time + warmup_time,
        **perf,
    }


def _strategy_parallel_submodule(model, device, input_shape):
    raw = get_raw_model(model)
    submod_names = ['backbone']
    if hasattr(raw, 'head_interaction'):
        submod_names.append('head_interaction')
    if hasattr(raw, 'pre_head_comm'):
        submod_names.append('pre_head_comm')

    compile_start = time.time()
    compiled_count = 0
    for name in submod_names:
        submod = getattr(raw, name, None)
        if submod is None:
            continue
        try:
            compiled_submod = t.compile(submod, mode='default')
            setattr(raw, name, compiled_submod)
            compiled_count += 1
        except Exception:
            pass
    compile_time = time.time() - compile_start

    if compiled_count == 0:
        return {'status': 'fail', 'error': 'no submodules compiled'}

    warmup_start = time.time()
    warmup_model(model, device, [input_shape], num_steps=2)
    warmup_time = time.time() - warmup_start

    perf = _measure_perf(model, device, input_shape)
    return {
        'status': 'pass',
        'compile_time_s': compile_time,
        'warmup_time_s': warmup_time,
        'total_setup_time_s': compile_time + warmup_time,
        'compiled_submodules': submod_names[:compiled_count],
        **perf,
    }


def _strategy_shape_hint(model, device, input_shape):
    compile_start = time.time()
    model, ok = try_compile_model(model, mode='default', dynamic=True)
    compile_time = time.time() - compile_start
    if not ok:
        return {'status': 'fail', 'error': 'compile failed'}

    bs = input_shape[0]
    hint_shapes = []
    for h, w in INPUT_SIZES:
        hint_shapes.append((bs, h, w))
    hint_shapes.append(input_shape)

    warmup_start = time.time()
    for shape in hint_shapes:
        try:
            dummy = _make_dummy(shape, device)
            with t.no_grad():
                _ = model(dummy)
            if t.cuda.is_available():
                t.cuda.synchronize()
            del dummy
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                t.cuda.empty_cache()
            else:
                raise
    warmup_time = time.time() - warmup_start

    perf = _measure_perf(model, device, input_shape)
    return {
        'status': 'pass',
        'compile_time_s': compile_time,
        'warmup_time_s': warmup_time,
        'total_setup_time_s': compile_time + warmup_time,
        'num_hint_shapes': len(hint_shapes),
        **perf,
    }


def _strategy_mixed_eager_compile(model, device, input_shape):
    raw = get_raw_model(model)
    if hasattr(raw, 'backbone'):
        compile_start = time.time()
        try:
            raw.backbone = t.compile(raw.backbone, mode='default')
            compile_time = time.time() - compile_start
        except Exception as e:
            return {'status': 'fail', 'error': f'backbone compile failed: {e}'}
    else:
        return {'status': 'fail', 'error': 'no backbone attribute'}

    warmup_start = time.time()
    warmup_model(model, device, [input_shape], num_steps=2)
    warmup_time = time.time() - warmup_start

    perf = _measure_perf(model, device, input_shape)
    return {
        'status': 'pass',
        'compile_time_s': compile_time,
        'warmup_time_s': warmup_time,
        'total_setup_time_s': compile_time + warmup_time,
        'compiled_parts': ['backbone'],
        'eager_parts': ['heads', 'head_interaction', 'roi'],
        **perf,
    }


def _strategy_reduce_overhead(model, device, input_shape):
    compile_start = time.time()
    model, ok = try_compile_model(model, mode='reduce-overhead', dynamic=True)
    compile_time = time.time() - compile_start
    if not ok:
        return {'status': 'fail', 'error': 'compile failed'}

    warmup_start = time.time()
    warmup_model(model, device, [input_shape], num_steps=2)
    warmup_time = time.time() - warmup_start

    perf = _measure_perf(model, device, input_shape)
    return {
        'status': 'pass',
        'compile_time_s': compile_time,
        'warmup_time_s': warmup_time,
        'total_setup_time_s': compile_time + warmup_time,
        **perf,
    }


def _strategy_fullgraph(model, device, input_shape):
    compile_start = time.time()
    model, ok = try_compile_model(model, mode='default', dynamic=False, fullgraph=True)
    compile_time = time.time() - compile_start
    if not ok:
        return {'status': 'fail', 'error': 'compile failed'}

    warmup_start = time.time()
    warmup_model(model, device, [input_shape], num_steps=2)
    warmup_time = time.time() - warmup_start

    perf = _measure_perf(model, device, input_shape)
    return {
        'status': 'pass',
        'compile_time_s': compile_time,
        'warmup_time_s': warmup_time,
        'total_setup_time_s': compile_time + warmup_time,
        **perf,
    }


def _strategy_inductor_config_tuned(model, device, input_shape):
    try:
        import torch._inductor.config as inductor_config
        inductor_config.triton.unique_kernel_names = True
        inductor_config.fx_graph_cache = True
        if hasattr(inductor_config, 'size_asserts'):
            inductor_config.size_asserts = False
    except Exception:
        pass

    configure_compile_cache()
    configure_dynamo_cache(cache_size_limit=512, accumulated_cache_size_limit=1024)

    compile_start = time.time()
    model, ok = try_compile_model(model, mode='default', dynamic=True)
    compile_time = time.time() - compile_start
    if not ok:
        return {'status': 'fail', 'error': 'compile failed'}

    warmup_start = time.time()
    warmup_model(model, device, [input_shape], num_steps=2)
    warmup_time = time.time() - warmup_start

    perf = _measure_perf(model, device, input_shape)
    return {
        'status': 'pass',
        'compile_time_s': compile_time,
        'warmup_time_s': warmup_time,
        'total_setup_time_s': compile_time + warmup_time,
        **perf,
    }


def audit_eager_baseline(recorder, device):
    print('\n' + '=' * 80)
    print('AUDIT 0: Eager Mode Baseline')
    print('=' * 80)

    for model_type in ALL_MODEL_TYPES:
        test_name = f'{model_type}_eager'
        print(f'  Testing {test_name}...', end=' ', flush=True)
        _reset_gpu_stats()

        try:
            model = _create_model(model_type, device)
            input_shape = (min(config.batch_size, 16), config.input_height, config.input_width)
            perf = _measure_perf(model, device, input_shape)
            perf['status'] = 'pass'
            perf['compile_time_s'] = 0
            perf['warmup_time_s'] = 0
            perf['total_setup_time_s'] = 0
            recorder.record('eager_baseline', test_name, perf)
            print(f"pass avg={perf['avg_latency_ms']:.2f}ms")
        except Exception as e:
            recorder.record('eager_baseline', test_name, {'status': 'fail', 'error': str(e)})
            print(f'fail: {e}')
        finally:
            try:
                del model
            except Exception:
                pass
            _reset_gpu_stats()


def audit_resource_usage(recorder, device):
    print('\n' + '=' * 80)
    print('AUDIT 9: Resource Usage (GPU Memory / Compile Overhead)')
    print('=' * 80)

    model_type = 'fpn_multihead'
    input_shape = (min(config.batch_size, 16), config.input_height, config.input_width)

    configs_to_test = [
        {'name': 'eager', 'compile': False},
        {'name': 'default_dynamic', 'compile': True, 'mode': 'default', 'dynamic': True},
        {'name': 'default_static', 'compile': True, 'mode': 'default', 'dynamic': False},
        {'name': 'reduce_overhead', 'compile': True, 'mode': 'reduce-overhead', 'dynamic': True},
        {'name': 'max_autotune', 'compile': True, 'mode': 'max-autotune', 'dynamic': True},
        {'name': 'fullgraph', 'compile': True, 'mode': 'default', 'dynamic': False, 'fullgraph': True},
        {'name': 'selective_backbone', 'compile': 'selective', 'submodules': ['backbone']},
    ]

    for cfg in configs_to_test:
        test_name = cfg['name']
        print(f'  Testing {test_name}...', end=' ', flush=True)
        _reset_gpu_stats()

        try:
            model = _create_model(model_type, device)
            param_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)

            compile_start = time.time()
            if cfg['compile'] is True:
                model, ok = try_compile_model(
                    model, mode=cfg.get('mode', 'default'),
                    dynamic=cfg.get('dynamic', True),
                    fullgraph=cfg.get('fullgraph', False))
                if not ok:
                    recorder.record('resource_usage', test_name,
                                    {'status': 'fail', 'error': 'compile failed'})
                    print('compile failed')
                    del model
                    continue
            elif cfg['compile'] == 'selective':
                model, ok = selective_compile(model, mode='default',
                                              compile_submodules=cfg.get('submodules', ['backbone']))
                if not ok:
                    recorder.record('resource_usage', test_name,
                                    {'status': 'fail', 'error': 'selective compile failed'})
                    print('selective compile failed')
                    del model
                    continue
            compile_time = time.time() - compile_start

            warmup_start = time.time()
            warmup_model(model, device, [input_shape], num_steps=2)
            warmup_time = time.time() - warmup_start

            alloc_gb, peak_gb = _gpu_mem_gb()

            result = {
                'status': 'pass',
                'param_mem_gb': param_mem,
                'compile_time_s': compile_time,
                'warmup_time_s': warmup_time,
                'total_setup_time_s': compile_time + warmup_time,
                'gpu_mem_after_compile_gb': alloc_gb,
                'gpu_mem_peak_gb': peak_gb,
                'compile_overhead_gb': peak_gb - param_mem,
            }
            recorder.record('resource_usage', test_name, result)
            print(f"pass param={param_mem:.3f}GB peak={peak_gb:.2f}GB overhead={result['compile_overhead_gb']:.3f}GB")
        except Exception as e:
            recorder.record('resource_usage', test_name, {'status': 'fail', 'error': str(e)})
            print(f'fail: {e}')
        finally:
            try:
                del model
            except Exception:
                pass
            _reset_gpu_stats()


def audit_compile_reproducibility(recorder, device):
    print('\n' + '=' * 80)
    print('AUDIT 10: Compile Reproducibility (Same Input -> Same Output)')
    print('=' * 80)

    model_type = 'fpn_multihead'
    mode = 'default'

    for model_type_cur in ALL_MODEL_TYPES:
        test_name = f'{model_type_cur}_reproducibility'
        print(f'  Testing {test_name}...', end=' ', flush=True)
        _reset_gpu_stats()

        try:
            model = _create_model(model_type_cur, device)
            model, ok = try_compile_model(model, mode=mode, dynamic=True)
            if not ok:
                recorder.record('reproducibility', test_name,
                                {'status': 'fail', 'error': 'compile failed'})
                print('compile failed')
                del model
                continue

            input_shape = (4, config.input_height, config.input_width)
            warmup_model(model, device, [input_shape], num_steps=2)

            dummy = _make_dummy(input_shape, device)
            with t.no_grad():
                out1 = model(dummy)
                out2 = model(dummy)
            del dummy

            reproducible = True
            max_diff = 0.0
            if isinstance(out1, (tuple, list)):
                for o1, o2 in zip(out1, out2):
                    if isinstance(o1, t.Tensor) and isinstance(o2, t.Tensor):
                        diff = (o1.detach().cpu().float() - o2.detach().cpu().float()).abs().max().item()
                        max_diff = max(max_diff, diff)
                        if diff > 1e-6:
                            reproducible = False
            elif isinstance(out1, t.Tensor):
                diff = (out1.detach().cpu().float() - out2.detach().cpu().float()).abs().max().item()
                max_diff = diff
                if diff > 1e-6:
                    reproducible = False

            result = {
                'status': 'pass' if reproducible else 'irreproducible',
                'reproducible': reproducible,
                'max_diff': max_diff,
            }
            recorder.record('reproducibility', test_name, result)
            print(f"{'pass' if reproducible else 'IRREPRODUCIBLE'} max_diff={max_diff:.10f}")
        except Exception as e:
            recorder.record('reproducibility', test_name, {'status': 'fail', 'error': str(e)})
            print(f'fail: {e}')
        finally:
            try:
                del model
            except Exception:
                pass
            _reset_gpu_stats()


def audit_compile_cache_hit(recorder, device):
    print('\n' + '=' * 80)
    print('AUDIT 11: Compile Cache Behavior (First vs Second Run)')
    print('=' * 80)

    model_type = 'fpn_multihead'
    mode = 'default'
    input_shape = (min(config.batch_size, 16), config.input_height, config.input_width)

    for dynamic in [True, False]:
        test_name = f'dynamic={dynamic}'
        print(f'  Testing {test_name}...', end=' ', flush=True)

        times = []
        for run_idx in range(2):
            _reset_gpu_stats()
            try:
                model = _create_model(model_type, device)
                compile_start = time.time()
                model, ok = try_compile_model(model, mode=mode, dynamic=dynamic)
                compile_time = time.time() - compile_start

                if not ok:
                    times.append({'run': run_idx, 'status': 'compile_failed'})
                    del model
                    continue

                warmup_start = time.time()
                warmup_model(model, device, [input_shape], num_steps=2)
                warmup_time = time.time() - warmup_start

                perf = _measure_perf(model, device, input_shape)
                times.append({
                    'run': run_idx,
                    'status': 'pass',
                    'compile_time_s': compile_time,
                    'warmup_time_s': warmup_time,
                    'total_setup_time_s': compile_time + warmup_time,
                    **perf,
                })
            except Exception as e:
                times.append({'run': run_idx, 'status': 'fail', 'error': str(e)})
            finally:
                try:
                    del model
                except Exception:
                    pass
                _reset_gpu_stats()

        result = {'runs': times}
        if len(times) == 2 and times[0].get('status') == 'pass' and times[1].get('status') == 'pass':
            speedup = times[0].get('total_setup_time_s', 1) / max(times[1].get('total_setup_time_s', 0.001), 0.001)
            result['setup_speedup'] = speedup
            print(f"run0={times[0]['total_setup_time_s']:.1f}s run1={times[1]['total_setup_time_s']:.1f}s speedup={speedup:.2f}x")
        else:
            print('incomplete')

        recorder.record('cache_behavior', test_name, result)


def generate_report(recorder):
    print('\n' + '=' * 80)
    print('COMPREHENSIVE ANALYSIS REPORT')
    print('=' * 80)

    eager_baselines = recorder.results.get('eager_baseline', {})
    eager_avg = {}
    for name, data in eager_baselines.items():
        if data.get('status') == 'pass':
            model_type = name.replace('_eager', '')
            eager_avg[model_type] = data.get('avg_latency_ms', float('inf'))

    warmup_results = recorder.results.get('warmup_strategies', {})
    if warmup_results:
        print('\n--- Warmup Strategy Ranking (by avg_latency_ms) ---')
        ranked = []
        for name, data in warmup_results.items():
            if data.get('status') == 'pass':
                ranked.append((name, data.get('avg_latency_ms', float('inf')),
                               data.get('total_setup_time_s', float('inf')),
                               data.get('first_latency_s', float('inf'))))
        ranked.sort(key=lambda x: x[1])
        for i, (name, avg_lat, setup, first) in enumerate(ranked):
            baseline = eager_avg.get('fpn_multihead', float('inf'))
            speedup = baseline / avg_lat if avg_lat > 0 else 0
            print(f'  {i+1}. {name}: avg={avg_lat:.2f}ms setup={setup:.1f}s '
                  f'first={first:.3f}s speedup={speedup:.2f}x vs eager')

    mode_results = recorder.results.get('compile_modes', {})
    if mode_results:
        print('\n--- Compile Mode Ranking (by avg_latency_ms, fpn_multihead) ---')
        ranked = []
        for name, data in mode_results.items():
            if name.startswith('fpn_multihead') and data.get('status') == 'pass':
                ranked.append((name, data.get('avg_latency_ms', float('inf')),
                               data.get('total_setup_time_s', float('inf'))))
        ranked.sort(key=lambda x: x[1])
        for i, (name, avg_lat, setup) in enumerate(ranked):
            print(f'  {i+1}. {name}: avg={avg_lat:.2f}ms setup={setup:.1f}s')

    resource_results = recorder.results.get('resource_usage', {})
    if resource_results:
        print('\n--- Resource Usage Comparison ---')
        for name, data in resource_results.items():
            if data.get('status') == 'pass':
                print(f'  {name}: param={data.get("param_mem_gb", 0):.3f}GB '
                      f'peak={data.get("gpu_mem_peak_gb", 0):.2f}GB '
                      f'overhead={data.get("compile_overhead_gb", 0):.3f}GB '
                      f'setup={data.get("total_setup_time_s", 0):.1f}s')

    correct_results = recorder.results.get('correctness', {})
    if correct_results:
        print('\n--- Correctness Summary ---')
        all_correct = True
        for name, data in correct_results.items():
            correct = data.get('correct', False)
            max_diff = data.get('max_diff', float('inf'))
            status = 'OK' if correct else 'MISMATCH'
            if not correct:
                all_correct = False
            print(f'  {name}: {status} max_diff={max_diff:.6f}')
        if all_correct:
            print('  >>> All compiled models produce correct outputs <<<')
        else:
            print('  >>> WARNING: Some compiled models produce incorrect outputs! <<<')

    print('\n--- Recommendations ---')
    best_strategy = None
    best_latency = float('inf')
    best_setup = None
    best_setup_time = float('inf')

    for name, data in warmup_results.items():
        if data.get('status') == 'pass':
            avg = data.get('avg_latency_ms', float('inf'))
            setup = data.get('total_setup_time_s', float('inf'))
            if avg < best_latency:
                best_latency = avg
                best_strategy = name
            if setup < best_setup_time:
                best_setup_time = setup
                best_setup = name

    if best_strategy:
        baseline = eager_avg.get('fpn_multihead', float('inf'))
        speedup = baseline / best_latency if best_latency > 0 else 0
        print(f'  Best inference performance: {best_strategy} '
              f'(avg={best_latency:.2f}ms, {speedup:.2f}x vs eager)')
    if best_setup:
        print(f'  Fastest setup time: {best_setup} ({best_setup_time:.1f}s)')

    dtype_results = recorder.results.get('input_dtypes', {})
    if dtype_results:
        print('\n--- Data Type Compatibility ---')
        for name, data in dtype_results.items():
            status = data.get('status', 'unknown')
            if status == 'pass':
                print(f'  {name}: COMPATIBLE avg={data.get("avg_latency_ms", 0):.2f}ms')
            else:
                print(f'  {name}: INCOMPATIBLE ({data.get("error", "unknown")[:60]})')

    dyn_results = recorder.results.get('dynamic_vs_static', {})
    if dyn_results:
        print('\n--- Dynamic vs Static Shape ---')
        for name, data in dyn_results.items():
            status = data.get('status', 'unknown')
            if status == 'pass':
                print(f'  {name}: avg={data.get("avg_latency_ms", 0):.2f}ms '
                      f'setup={data.get("total_setup_time_s", 0):.1f}s')
            else:
                print(f'  {name}: {status}')

    cache_results = recorder.results.get('cache_behavior', {})
    if cache_results:
        print('\n--- Cache Behavior ---')
        for name, data in cache_results.items():
            runs = data.get('runs', [])
            speedup = data.get('setup_speedup', None)
            if len(runs) == 2 and runs[0].get('status') == 'pass' and runs[1].get('status') == 'pass':
                print(f'  {name}: run0={runs[0]["total_setup_time_s"]:.1f}s '
                      f'run1={runs[1]["total_setup_time_s"]:.1f}s '
                      f'cache_speedup={speedup:.2f}x' if speedup else '')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='torch.compile Systematic Audit')
    parser.add_argument('--audit', type=str, default='all',
                        choices=['all', 'baseline', 'modes', 'dtypes', 'sizes',
                                 'batches', 'dynamic', 'models', 'correctness',
                                 'warmup', 'resource', 'reproducibility', 'cache'],
                        help='Which audit to run')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audit_results')
    else:
        output_dir = args.output_dir

    recorder = AuditRecorder(output_dir)

    print('=' * 80)
    print('torch.compile SYSTEMATIC AUDIT')
    print('=' * 80)
    print(f'Device: {device}')
    print(f'GPU Platform: {GPU_PLATFORM}')
    print(f'Total VRAM: {TOTAL_VRAM_GB:.1f} GB')
    print(f'Compile Available: {COMPILE_AVAILABLE}')
    print(f'PyTorch Version: {t.__version__}')
    if t.cuda.is_available():
        print(f'GPU: {t.cuda.get_device_name(0)}')
        props = t.cuda.get_device_properties(0)
        vram = getattr(props, 'total_mem', getattr(props, 'total_memory', 0)) / (1024**3)
        print(f'VRAM: {vram:.1f} GB')
    print(f'Python: {sys.version}')
    print(f'Config: batch_size={config.batch_size}, input={config.input_height}x{config.input_width}')
    print(f'Compile: mode={config.compile_mode}, dynamic={config.compile_dynamic}, fullgraph={config.compile_fullgraph}')
    print()

    if not COMPILE_AVAILABLE:
        print('[WARNING] torch.compile is NOT available on this platform!')
        print('  Running limited audit (eager baseline only)...')
        audit_eager_baseline(recorder, device)
        recorder.print_summary()
        recorder.save()
        return

    configure_compile_cache()

    audit_map = {
        'baseline': lambda: audit_eager_baseline(recorder, device),
        'modes': lambda: audit_compile_modes(recorder, device),
        'dtypes': lambda: audit_input_dtypes(recorder, device),
        'sizes': lambda: audit_input_sizes(recorder, device),
        'batches': lambda: audit_batch_sizes(recorder, device),
        'dynamic': lambda: audit_dynamic_vs_static(recorder, device),
        'models': lambda: audit_model_types(recorder, device),
        'correctness': lambda: audit_correctness(recorder, device),
        'warmup': lambda: audit_warmup_strategies(recorder, device),
        'resource': lambda: audit_resource_usage(recorder, device),
        'reproducibility': lambda: audit_compile_reproducibility(recorder, device),
        'cache': lambda: audit_compile_cache_hit(recorder, device),
    }

    if args.audit == 'all':
        order = ['baseline', 'modes', 'dtypes', 'sizes', 'batches', 'dynamic',
                 'models', 'correctness', 'warmup', 'resource', 'reproducibility', 'cache']
        for audit_name in order:
            try:
                audit_map[audit_name]()
            except Exception as e:
                print(f'\n[AUDIT ERROR] {audit_name} failed: {e}')
                traceback.print_exc()
    else:
        audit_map[args.audit]()

    generate_report(recorder)
    recorder.print_summary()
    path = recorder.save()

    print(f'\nAudit complete. Results at: {path}')


if __name__ == '__main__':
    main()
