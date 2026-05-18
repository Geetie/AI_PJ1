import os
import sys
import time
import json
import gc
import argparse
import numpy as np
import torch as t
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from models import create_model
from models.ctc import CTCModel
from utils.compile_utils import (
    is_compile_available, try_compile_model, warmup_model,
    get_raw_model, measure_compile_performance, configure_dynamo_cache,
    CompileLogger, configure_compile_cache, WARMUP_STRATEGIES, run_compile_audit
)


def benchmark_compile_modes(device, model_type='fpn_multihead'):
    results = []
    modes = ['default']
    if is_compile_available():
        modes.extend(['reduce-overhead', 'max-autotune'])

    for mode in modes:
        print(f'\n{"="*60}')
        print(f'Benchmarking compile_mode={mode}, model_type={model_type}')
        print(f'{"="*60}')

        gc.collect()
        t.cuda.empty_cache()
        if t.cuda.is_available():
            t.cuda.reset_peak_memory_stats()

        model = create_model(model_type).to(device)
        model.eval()

        compile_start = time.time()
        if mode != 'eager':
            model, ok = try_compile_model(model, mode=mode, dynamic=True)
            if not ok:
                print(f'  Compile failed for mode={mode}, skipping')
                del model
                gc.collect()
                t.cuda.empty_cache()
                continue
        compile_time = time.time() - compile_start

        input_shape = (config.batch_size, config.input_height, config.input_width)
        perf = measure_compile_performance(model, device, input_shape, num_warmup=3, num_iters=20)

        result = {
            'model_type': model_type,
            'compile_mode': mode,
            'compile_time_s': compile_time,
            **perf,
        }
        results.append(result)
        print(f'  compile_time={compile_time:.1f}s, avg_latency={perf["avg_latency_ms"]:.2f}ms, '
              f'p50={perf["p50_latency_ms"]:.2f}ms, p99={perf["p99_latency_ms"]:.2f}ms, '
              f'gpu_peak={perf.get("gpu_mem_peak_gb", 0):.2f}GB')

        del model
        gc.collect()
        t.cuda.empty_cache()

    return results


def benchmark_input_shapes(device, mode='default', model_type='fpn_multihead'):
    results = []
    shapes = [
        (16, 224, 224), (16, 288, 288), (16, 320, 320),
        (16, 352, 352), (16, 384, 384), (16, 416, 416),
        (32, 384, 384), (64, 384, 384),
        (1, 384, 384),
    ]

    model = create_model(model_type).to(device)
    model.eval()
    if is_compile_available():
        model, ok = try_compile_model(model, mode=mode, dynamic=True)
        if not ok:
            print('Compile failed, using eager mode')

    for bs, h, w in shapes:
        print(f'  Shape ({bs}, 3, {h}, {w})...', end=' ', flush=True)
        try:
            perf = measure_compile_performance(model, device, (bs, h, w),
                                               num_warmup=2, num_iters=10)
            result = {
                'model_type': model_type,
                'compile_mode': mode,
                'batch_size': bs,
                'height': h,
                'width': w,
                **perf,
            }
            results.append(result)
            print(f'avg={perf["avg_latency_ms"]:.2f}ms, first={perf["first_latency_s"]:.3f}s')
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print('OOM, skipping')
                t.cuda.empty_cache()
            else:
                raise

    del model
    gc.collect()
    t.cuda.empty_cache()
    return results


def benchmark_warmup_strategies(device, model_type='fpn_multihead'):
    results = []

    strategies = [
        {'name': 'no_warmup', 'warmup': False},
        {'name': 'single_shape', 'warmup': True, 'shapes': 'single'},
        {'name': 'multi_shape_tta', 'warmup': True, 'shapes': 'tta'},
    ]

    for strategy in strategies:
        print(f'\n  Strategy: {strategy["name"]}')
        gc.collect()
        t.cuda.empty_cache()
        if t.cuda.is_available():
            t.cuda.reset_peak_memory_stats()

        model = create_model(model_type).to(device)
        model.eval()

        compile_start = time.time()
        model, ok = try_compile_model(model, mode='default', dynamic=True)
        compile_time = time.time() - compile_start

        if not ok:
            del model
            continue

        warmup_time = 0
        if strategy['warmup']:
            if strategy['shapes'] == 'single':
                shapes = [(config.batch_size, config.input_height, config.input_width)]
            elif strategy['shapes'] == 'tta':
                shapes = [(config.batch_size, s, s) for s in config.tta_sizes]
                shapes.append((config.batch_size, config.input_height, config.input_width))
            else:
                shapes = [(config.batch_size, config.input_height, config.input_width)]

            wt_start = time.time()
            warmup_model(model, device, shapes, num_steps=1, use_amp=True)
            warmup_time = time.time() - wt_start

        input_shape = (config.batch_size, config.input_height, config.input_width)
        perf = measure_compile_performance(model, device, input_shape,
                                           num_warmup=3, num_iters=20)

        result = {
            'strategy': strategy['name'],
            'compile_time_s': compile_time,
            'warmup_time_s': warmup_time,
            'total_setup_time_s': compile_time + warmup_time,
            **perf,
        }
        results.append(result)
        print(f'    setup={compile_time + warmup_time:.1f}s, '
              f'avg_latency={perf["avg_latency_ms"]:.2f}ms')

        del model
        gc.collect()
        t.cuda.empty_cache()

    return results


def benchmark_selective_compile(device, model_type='fpn_multihead'):
    results = []

    configs = [
        {'name': 'full_model', 'submodules': None},
        {'name': 'backbone_only', 'submodules': ['backbone']},
    ]

    for cfg in configs:
        print(f'\n  Config: {cfg["name"]}')
        gc.collect()
        t.cuda.empty_cache()

        model = create_model(model_type).to(device)
        model.eval()

        compile_start = time.time()
        if cfg['submodules'] is None:
            model, ok = try_compile_model(model, mode='default', dynamic=True)
        else:
            from utils.compile_utils import selective_compile
            model, ok = selective_compile(model, mode='default',
                                          compile_submodules=cfg['submodules'])
        compile_time = time.time() - compile_start

        if not ok:
            del model
            continue

        input_shape = (config.batch_size, config.input_height, config.input_width)
        perf = measure_compile_performance(model, device, input_shape,
                                           num_warmup=3, num_iters=20)

        result = {
            'config': cfg['name'],
            'compile_time_s': compile_time,
            **perf,
        }
        results.append(result)
        print(f'    compile={compile_time:.1f}s, avg_latency={perf["avg_latency_ms"]:.2f}ms')

        del model
        gc.collect()
        t.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description='torch.compile Benchmark')
    parser.add_argument('--benchmark', type=str, default='all',
                        choices=['all', 'modes', 'shapes', 'warmup', 'selective', 'audit'],
                        help='Which benchmark to run')
    parser.add_argument('--model-type', type=str, default=None,
                        help='Model type to benchmark')
    args = parser.parse_args()

    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
    model_type = args.model_type or config.model_type

    compile_logger = CompileLogger.get_instance()

    print(f'torch.compile Benchmark')
    print(f'Device: {device}')
    print(f'Model type: {model_type}')
    print(f'Compile available: {is_compile_available()}')
    print(f'PyTorch version: {t.__version__}')
    if t.cuda.is_available():
        print(f'GPU: {t.cuda.get_device_name(0)}')
        props = t.cuda.get_device_properties(0)
        vram = getattr(props, 'total_mem', getattr(props, 'total_memory', 0)) / (1024**3)
        print(f'VRAM: {vram:.1f} GB')

    configure_compile_cache()

    all_results = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark_results')
    os.makedirs(output_dir, exist_ok=True)

    if args.benchmark in ('all', 'modes'):
        print('\n' + '='*80)
        print('BENCHMARK 1: Compile Modes')
        print('='*80)
        all_results['compile_modes'] = benchmark_compile_modes(device, model_type)

    if args.benchmark in ('all', 'shapes'):
        print('\n' + '='*80)
        print('BENCHMARK 2: Input Shapes')
        print('='*80)
        all_results['input_shapes'] = benchmark_input_shapes(device, model_type=model_type)

    if args.benchmark in ('all', 'warmup'):
        print('\n' + '='*80)
        print('BENCHMARK 3: Warmup Strategies')
        print('='*80)
        all_results['warmup_strategies'] = benchmark_warmup_strategies(device, model_type)

    if args.benchmark in ('all', 'selective'):
        print('\n' + '='*80)
        print('BENCHMARK 4: Selective Compilation')
        print('='*80)
        all_results['selective_compile'] = benchmark_selective_compile(device, model_type)

    if args.benchmark in ('all', 'audit'):
        print('\n' + '='*80)
        print('BENCHMARK 5: Full Compile Audit (All Strategies)')
        print('='*80)
        model = create_model(model_type).to(device)
        model.eval()
        audit_results, audit_summary = run_compile_audit(model, device)
        all_results['audit'] = audit_results
        all_results['audit_summary'] = audit_summary
        del model
        gc.collect()
        t.cuda.empty_cache()

    output_path = os.path.join(output_dir, f'compile_benchmark_{timestamp}.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nResults saved to {output_path}')

    compile_summary = compile_logger.print_summary()
    all_results['compile_logger_summary'] = compile_summary

    print('\n' + '='*80)
    print('SUMMARY')
    print('='*80)
    for bench_name, bench_results in all_results.items():
        print(f'\n--- {bench_name} ---')
        if isinstance(bench_results, list):
            for r in bench_results:
                print(f'  {r}')
        else:
            print(f'  {bench_results}')


if __name__ == '__main__':
    main()
