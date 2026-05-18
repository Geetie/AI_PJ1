import os
import sys
import time
import json
import threading
import logging
import traceback
from datetime import datetime
from contextlib import contextmanager

import torch as t


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)


def is_triton_available():
    if sys.platform == 'win32':
        return False
    try:
        import triton
        return True
    except (ImportError, OSError):
        return False


def is_compile_available():
    if not t.cuda.is_available():
        return False
    if sys.platform == 'win32':
        return False
    try:
        import torch._inductor
        return True
    except (ImportError, OSError):
        return False


def configure_dynamo_cache(cache_size_limit=256, accumulated_cache_size_limit=512):
    try:
        import torch._dynamo as dynamo
        dynamo.config.cache_size_limit = cache_size_limit
        dynamo.config.accumulated_cache_size_limit = accumulated_cache_size_limit
    except Exception:
        pass


class CompileLogger:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, log_dir=None):
        if log_dir is None:
            log_dir = os.path.join(_PROJECT_DIR, 'logs', 'compile')
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path = os.path.join(log_dir, f'compile_{timestamp}.log')
        self.json_log_path = os.path.join(log_dir, f'compile_{timestamp}.json')

        self.logger = logging.getLogger(f'CompileLogger_{timestamp}')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []

        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        fh = logging.FileHandler(self.log_path, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

        self._events = []
        self._compile_start_time = None
        self._warmup_events = []
        self._phase_stack = []

    @classmethod
    def get_instance(cls, log_dir=None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(log_dir)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        with cls._lock:
            if cls._instance is not None:
                for h in cls._instance.logger.handlers:
                    h.close()
                cls._instance = None

    def _record_event(self, event_type, data):
        event = {
            'timestamp': time.time(),
            'event_type': event_type,
            **data,
        }
        if t.cuda.is_available():
            event['gpu_mem_allocated_gb'] = t.cuda.memory_allocated() / (1024**3)
            event['gpu_mem_reserved_gb'] = t.cuda.memory_reserved() / (1024**3)
        self._events.append(event)
        return event

    def log_compile_start(self, model_type, mode='default', dynamic=False,
                          fullgraph=False, input_shape=None, backend='inductor'):
        self._compile_start_time = time.time()
        data = {
            'model_type': model_type,
            'mode': mode,
            'dynamic': dynamic,
            'fullgraph': fullgraph,
            'input_shape': list(input_shape) if input_shape else None,
            'backend': backend,
            'torch_version': t.__version__,
            'cuda_version': t.version.cuda if t.cuda.is_available() else None,
        }
        if t.cuda.is_available():
            data['gpu_name'] = t.cuda.get_device_name(0)
            props = t.cuda.get_device_properties(0)
            data['gpu_vram_gb'] = getattr(
                props, 'total_mem', getattr(props, 'total_memory', 0)) / (1024**3)
        self._record_event('compile_start', data)
        self.logger.info(
            f'[COMPILE-START] model={model_type} mode={mode} dynamic={dynamic} '
            f'fullgraph={fullgraph} backend={backend}')
        if input_shape:
            self.logger.info(f'[COMPILE-START] input_shape={input_shape}')

    def log_compile_end(self, success, compile_time_s=None, error=None):
        if compile_time_s is None and self._compile_start_time is not None:
            compile_time_s = time.time() - self._compile_start_time
        data = {
            'success': success,
            'compile_time_s': compile_time_s,
        }
        if error is not None:
            data['error'] = str(error)
            data['error_type'] = type(error).__name__
            data['traceback'] = traceback.format_exc()
        if t.cuda.is_available():
            data['gpu_mem_peak_gb'] = t.cuda.max_memory_allocated() / (1024**3)
        self._compile_start_time = None
        self._record_event('compile_end', data)
        status = 'OK' if success else 'FAILED'
        self.logger.info(
            f'[COMPILE-END] {status} time={compile_time_s:.2f}s'
            + (f' error={error}' if error else ''))

    def log_warmup_shape(self, shape, step, latency_s, is_first_compile=False,
                         oom=False, error=None):
        data = {
            'shape': list(shape),
            'step': step,
            'latency_s': latency_s,
            'is_first_compile': is_first_compile,
            'oom': oom,
        }
        if error is not None:
            data['error'] = str(error)
        self._warmup_events.append(data)
        self._record_event('warmup_shape', data)
        tag = ' [FIRST-COMPILE]' if is_first_compile else ''
        oom_tag = ' [OOM]' if oom else ''
        self.logger.info(
            f'[WARMUP-SHAPE] shape={shape} step={step} latency={latency_s:.3f}s'
            f'{tag}{oom_tag}')

    def log_warmup_summary(self, total_time_s, num_shapes, num_steps,
                           shapes_compiled=None, cache_hits=None):
        data = {
            'total_time_s': total_time_s,
            'num_shapes': num_shapes,
            'num_steps': num_steps,
            'shapes_compiled': shapes_compiled,
            'cache_hits': cache_hits,
        }
        self._record_event('warmup_summary', data)
        self.logger.info(
            f'[WARMUP-SUMMARY] total={total_time_s:.1f}s shapes={num_shapes} '
            f'steps={num_steps}')

    def log_inference_perf(self, input_shape, first_latency_s, avg_latency_ms,
                           p50_latency_ms, p99_latency_ms, min_latency_ms,
                           max_latency_ms, gpu_mem_allocated_gb=None,
                           gpu_mem_peak_gb=None):
        data = {
            'input_shape': list(input_shape),
            'first_latency_s': first_latency_s,
            'avg_latency_ms': avg_latency_ms,
            'p50_latency_ms': p50_latency_ms,
            'p99_latency_ms': p99_latency_ms,
            'min_latency_ms': min_latency_ms,
            'max_latency_ms': max_latency_ms,
        }
        if gpu_mem_allocated_gb is not None:
            data['gpu_mem_allocated_gb'] = gpu_mem_allocated_gb
        if gpu_mem_peak_gb is not None:
            data['gpu_mem_peak_gb'] = gpu_mem_peak_gb
        self._record_event('inference_perf', data)
        self.logger.info(
            f'[INFER-PERF] shape={input_shape} first={first_latency_s:.3f}s '
            f'avg={avg_latency_ms:.2f}ms p50={p50_latency_ms:.2f}ms '
            f'p99={p99_latency_ms:.2f}ms')

    def log_dynamo_stats(self):
        try:
            import torch._dynamo as dynamo
            stats = dynamo.utils.compile_times()
            if stats:
                data = {'dynamo_compile_times': stats}
                self._record_event('dynamo_stats', data)
                self.logger.info(f'[DYNAMO-STATS] {stats}')
        except Exception:
            pass

    def log_cache_info(self):
        try:
            import torch._dynamo as dynamo
            cache_info = {
                'cache_size_limit': getattr(dynamo.config, 'cache_size_limit', None),
                'accumulated_cache_size_limit': getattr(
                    dynamo.config, 'accumulated_cache_size_limit', None),
            }
            data = {'dynamo_cache_config': cache_info}
            self._record_event('cache_info', data)
            self.logger.info(f'[CACHE-INFO] {cache_info}')
        except Exception:
            pass

    def log_compile_config(self, config_dict):
        self._record_event('compile_config', config_dict)
        self.logger.info(f'[COMPILE-CONFIG] {config_dict}')

    def log_phase_start(self, phase_name):
        data = {'phase': phase_name}
        self._record_event('phase_start', data)
        self._phase_stack.append((phase_name, time.time()))
        self.logger.info(f'[PHASE-START] {phase_name}')

    def log_phase_end(self, phase_name):
        duration = None
        if self._phase_stack and self._phase_stack[-1][0] == phase_name:
            _, start = self._phase_stack.pop()
            duration = time.time() - start
        data = {'phase': phase_name, 'duration_s': duration}
        self._record_event('phase_end', data)
        self.logger.info(
            f'[PHASE-END] {phase_name} duration={duration:.2f}s'
            if duration else f'[PHASE-END] {phase_name}')

    @contextmanager
    def phase(self, phase_name):
        self.log_phase_start(phase_name)
        try:
            yield
        finally:
            self.log_phase_end(phase_name)

    def log_strategy_result(self, strategy_name, compile_time_s, warmup_time_s,
                            total_setup_time_s, perf_data):
        data = {
            'strategy': strategy_name,
            'compile_time_s': compile_time_s,
            'warmup_time_s': warmup_time_s,
            'total_setup_time_s': total_setup_time_s,
            **perf_data,
        }
        self._record_event('strategy_result', data)
        self.logger.info(
            f'[STRATEGY] {strategy_name} setup={total_setup_time_s:.1f}s '
            f'compile={compile_time_s:.1f}s warmup={warmup_time_s:.1f}s '
            f'avg_latency={perf_data.get("avg_latency_ms", 0):.2f}ms')

    def save_json_log(self):
        try:
            with open(self.json_log_path, 'w', encoding='utf-8') as f:
                json.dump(self._events, f, indent=2, default=str)
            self.logger.info(f'[LOG] JSON log saved to {self.json_log_path}')
        except Exception as e:
            self.logger.warning(f'[LOG] Failed to save JSON log: {e}')

    def get_summary(self):
        compile_events = [e for e in self._events if e['event_type'] == 'compile_end']
        warmup_events = [e for e in self._events if e['event_type'] == 'warmup_shape']
        perf_events = [e for e in self._events if e['event_type'] == 'inference_perf']
        strategy_events = [e for e in self._events if e['event_type'] == 'strategy_result']

        summary = {
            'total_compile_attempts': len(compile_events),
            'successful_compiles': sum(1 for e in compile_events if e.get('success')),
            'failed_compiles': sum(1 for e in compile_events if not e.get('success')),
            'total_warmup_shapes': len(warmup_events),
            'warmup_first_compiles': sum(
                1 for e in warmup_events if e.get('is_first_compile')),
            'warmup_oom_events': sum(1 for e in warmup_events if e.get('oom')),
            'total_perf_measurements': len(perf_events),
            'strategies_tested': len(strategy_events),
        }

        if compile_events:
            times = [e['compile_time_s'] for e in compile_events
                     if e.get('compile_time_s') is not None]
            if times:
                summary['avg_compile_time_s'] = sum(times) / len(times)
                summary['max_compile_time_s'] = max(times)
                summary['min_compile_time_s'] = min(times)

        if perf_events:
            avg_latencies = [e['avg_latency_ms'] for e in perf_events
                             if e.get('avg_latency_ms') is not None]
            if avg_latencies:
                summary['avg_inference_latency_ms'] = sum(avg_latencies) / len(avg_latencies)

        return summary

    def print_summary(self):
        summary = self.get_summary()
        self.logger.info('=' * 60)
        self.logger.info('[COMPILE-SUMMARY]')
        for k, v in summary.items():
            self.logger.info(f'  {k}: {v}')
        self.logger.info('=' * 60)
        self.save_json_log()
        return summary


def _get_model_type_name(model):
    raw = model
    while hasattr(raw, '_orig_mod'):
        raw = raw._orig_mod
    return type(raw).__name__


def try_compile_model(model, mode='default', dynamic=False, fullgraph=False):
    logger = CompileLogger.get_instance()
    if not is_compile_available():
        logger.logger.info('[COMPILE] torch.compile not available on this platform, skipping')
        return model, False

    model_type = _get_model_type_name(model)
    logger.log_compile_start(model_type, mode=mode, dynamic=dynamic, fullgraph=fullgraph)
    logger.log_cache_info()

    try:
        compile_start = time.time()
        compiled = t.compile(model, mode=mode, dynamic=dynamic, fullgraph=fullgraph)
        compile_time = time.time() - compile_start
        logger.log_compile_end(success=True, compile_time_s=compile_time)
        logger.log_dynamo_stats()
        logger.logger.info(
            f'[COMPILE] torch.compile enabled (mode={mode}, dynamic={dynamic}, '
            f'fullgraph={fullgraph}) time={compile_time:.2f}s')
        return compiled, True
    except Exception as e:
        logger.log_compile_end(success=False, error=e)
        logger.logger.info(f'[COMPILE] torch.compile failed: {e}, falling back to eager mode')
        return model, False


def warmup_model(model, device, input_shapes, forward_fn=None, num_steps=1,
                 use_amp=True, timeout_seconds=600):
    logger = CompileLogger.get_instance()
    if not t.cuda.is_available():
        return 0.0

    heartbeat_stop = threading.Event()
    warmup_start = time.time()

    def _heartbeat():
        elapsed = 0
        while not heartbeat_stop.wait(30):
            elapsed += 30
            logger.logger.info(f'[WARMUP] Still compiling kernels... ({elapsed}s elapsed)')

    ht = threading.Thread(target=_heartbeat, daemon=True)
    ht.start()

    try:
        total_time = _run_warmup(model, device, input_shapes, forward_fn, num_steps, use_amp)
        warmup_time = time.time() - warmup_start
        logger.log_warmup_summary(warmup_time, len(input_shapes), num_steps)
        logger.log_dynamo_stats()
        logger.logger.info(
            f'[WARMUP] Completed in {warmup_time:.1f}s '
            f'({len(input_shapes)} shapes x {num_steps} steps)')
        return warmup_time
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            logger.logger.info(f'[WARMUP] OOM during warmup, trying reduced shapes...')
            t.cuda.empty_cache()
            reduced = input_shapes[:max(1, len(input_shapes) // 2)]
            try:
                _run_warmup(model, device, reduced, forward_fn, 1, use_amp)
                warmup_time = time.time() - warmup_start
                logger.log_warmup_summary(warmup_time, len(reduced), 1)
                logger.logger.info(f'[WARMUP] Reduced warmup completed in {warmup_time:.1f}s')
                return warmup_time
            except Exception:
                logger.logger.info(f'[WARMUP] Reduced warmup also failed')
                t.cuda.empty_cache()
                return time.time() - warmup_start
        else:
            logger.logger.info(f'[WARMUP] Warmup failed: {e}')
            t.cuda.empty_cache()
            return time.time() - warmup_start
    except Exception as e:
        err_str = str(e).lower()
        if 'compile' in err_str or 'triton' in err_str or 'inductor' in err_str:
            logger.logger.info(f'[WARMUP] torch.compile error during warmup: {e}')
            raise
        logger.logger.info(f'[WARMUP] Warmup failed: {e}')
        t.cuda.empty_cache()
        return time.time() - warmup_start
    finally:
        heartbeat_stop.set()
        ht.join(timeout=5)


def _run_warmup(model, device, input_shapes, forward_fn, num_steps, use_amp):
    from torch.amp import autocast
    logger = CompileLogger.get_instance()

    for shape_idx, shape_info in enumerate(input_shapes):
        bs, h, w = shape_info
        dummy = t.randn(bs, 3, h, w, device=device)
        torch_device = device if isinstance(device, t.device) else t.device(device)

        for step in range(num_steps):
            step_start = time.time()
            is_first = (shape_idx == 0 and step == 0)
            try:
                if forward_fn is not None:
                    with autocast(torch_device.type, enabled=use_amp):
                        forward_fn(model, dummy)
                else:
                    with t.no_grad(), autocast(torch_device.type, enabled=use_amp):
                        _ = model(dummy)
                step_latency = time.time() - step_start
                logger.log_warmup_shape(
                    shape_info, step, step_latency, is_first_compile=is_first)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    step_latency = time.time() - step_start
                    logger.log_warmup_shape(
                        shape_info, step, step_latency, is_first_compile=is_first, oom=True)
                    raise
                raise

        del dummy

    t.cuda.synchronize()
    t.cuda.empty_cache()


def warmup_multi_shape(model, device, shapes, batch_sizes, forward_fn=None,
                       use_amp=True):
    input_shapes = [(bs, h, w) for h, w in shapes for bs in batch_sizes]
    return warmup_model(model, device, input_shapes, forward_fn=forward_fn,
                        num_steps=1, use_amp=use_amp)


def selective_compile(model, mode='default', compile_submodules=None):
    logger = CompileLogger.get_instance()
    if not is_compile_available():
        logger.logger.info('[COMPILE] torch.compile not available, skipping selective compile')
        return model, False

    if compile_submodules is None:
        return try_compile_model(model, mode=mode)

    success = False
    for name in compile_submodules:
        submod = getattr(model, name, None)
        if submod is None:
            continue
        try:
            logger.log_compile_start(
                f'submodule:{name}', mode=mode, dynamic=False, fullgraph=False)
            compile_start = time.time()
            compiled_submod = t.compile(submod, mode=mode)
            setattr(model, name, compiled_submod)
            compile_time = time.time() - compile_start
            logger.log_compile_end(success=True, compile_time_s=compile_time)
            logger.logger.info(
                f'[COMPILE] Selectively compiled submodule: {name} (mode={mode}) '
                f'time={compile_time:.2f}s')
            success = True
        except Exception as e:
            logger.log_compile_end(success=False, error=e)
            logger.logger.info(f'[COMPILE] Failed to compile submodule {name}: {e}')

    return model, success


def get_raw_model(model):
    current = model
    while hasattr(current, '_orig_mod'):
        current = current._orig_mod
    return current


def measure_compile_performance(model, device, input_shape, num_warmup=3, num_iters=20,
                                use_amp=True, forward_fn=None):
    from torch.amp import autocast
    logger = CompileLogger.get_instance()

    results = {}
    bs, h, w = input_shape
    torch_device = device if isinstance(device, t.device) else t.device(device)

    dummy = t.randn(bs, 3, h, w, device=device)

    t0 = time.time()
    with t.no_grad(), autocast(torch_device.type, enabled=use_amp):
        if forward_fn:
            forward_fn(model, dummy)
        else:
            _ = model(dummy)
    t.cuda.synchronize()
    first_latency = time.time() - t0
    results['first_latency_s'] = first_latency

    for _ in range(num_warmup):
        with t.no_grad(), autocast(torch_device.type, enabled=use_amp):
            if forward_fn:
                forward_fn(model, dummy)
            else:
                _ = model(dummy)
    t.cuda.synchronize()

    latencies = []
    for _ in range(num_iters):
        t0 = time.time()
        with t.no_grad(), autocast(torch_device.type, enabled=use_amp):
            if forward_fn:
                forward_fn(model, dummy)
            else:
                _ = model(dummy)
        t.cuda.synchronize()
        latencies.append(time.time() - t0)

    import numpy as np
    latencies = np.array(latencies) * 1000
    results['avg_latency_ms'] = float(latencies.mean())
    results['p50_latency_ms'] = float(np.percentile(latencies, 50))
    results['p99_latency_ms'] = float(np.percentile(latencies, 99))
    results['min_latency_ms'] = float(latencies.min())
    results['max_latency_ms'] = float(latencies.max())

    if t.cuda.is_available():
        results['gpu_mem_allocated_gb'] = t.cuda.memory_allocated() / (1024**3)
        results['gpu_mem_peak_gb'] = t.cuda.max_memory_allocated() / (1024**3)
        t.cuda.reset_peak_memory_stats()

    logger.log_inference_perf(
        input_shape, first_latency,
        results['avg_latency_ms'], results['p50_latency_ms'],
        results['p99_latency_ms'], results['min_latency_ms'],
        results['max_latency_ms'],
        gpu_mem_allocated_gb=results.get('gpu_mem_allocated_gb'),
        gpu_mem_peak_gb=results.get('gpu_mem_peak_gb'))

    del dummy
    t.cuda.empty_cache()

    return results


def configure_compile_cache(cache_dir=None):
    logger = CompileLogger.get_instance()
    if cache_dir is None:
        cache_dir = os.path.join(_PROJECT_DIR, 'compile_cache')
    os.makedirs(cache_dir, exist_ok=True)

    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', cache_dir)
    os.environ.setdefault('XLA_CACHE_DIR', cache_dir)

    logger.log_compile_config({
        'TORCHINDUCTOR_CACHE_DIR': cache_dir,
        'TORCHINDUCTOR_FORCE_REDIRECT_CACHES': os.environ.get(
            'TORCHINDUCTOR_FORCE_REDIRECT_CACHES', '0'),
    })
    logger.logger.info(f'[CACHE] Compile cache directory: {cache_dir}')
    return cache_dir


def warmup_strategy_single(model, device, mode='default', dynamic=False,
                           input_shape=None, use_amp=True, forward_fn=None):
    logger = CompileLogger.get_instance()
    strategy_name = 'single_shape'

    if input_shape is None:
        from config import config as cfg
        input_shape = (cfg.batch_size, cfg.input_height, cfg.input_width)

    compile_start = time.time()
    model, ok = try_compile_model(model, mode=mode, dynamic=dynamic)
    compile_time = time.time() - compile_start
    if not ok:
        return model, {'strategy': strategy_name, 'success': False}

    with logger.phase(f'warmup_{strategy_name}'):
        wt_start = time.time()
        warmup_model(model, device, [input_shape], forward_fn=forward_fn,
                     num_steps=2, use_amp=use_amp)
        warmup_time = time.time() - wt_start

    perf = measure_compile_performance(model, device, input_shape,
                                       num_warmup=3, num_iters=20,
                                       use_amp=use_amp, forward_fn=forward_fn)
    result = {
        'strategy': strategy_name,
        'success': True,
        'compile_time_s': compile_time,
        'warmup_time_s': warmup_time,
        'total_setup_time_s': compile_time + warmup_time,
        **perf,
    }
    logger.log_strategy_result(strategy_name, compile_time, warmup_time,
                               compile_time + warmup_time, perf)
    return model, result


def warmup_strategy_tta(model, device, mode='default', dynamic=True,
                        tta_sizes=None, batch_size=None, use_amp=True,
                        forward_fn=None):
    logger = CompileLogger.get_instance()
    strategy_name = 'multi_shape_tta'

    if tta_sizes is None:
        from config import config as cfg
        tta_sizes = cfg.tta_sizes
    if batch_size is None:
        from config import config as cfg
        batch_size = min(cfg.eval_batch_size, cfg.batch_size)

    compile_start = time.time()
    model, ok = try_compile_model(model, mode=mode, dynamic=dynamic)
    compile_time = time.time() - compile_start
    if not ok:
        return model, {'strategy': strategy_name, 'success': False}

    from config import config as cfg
    shapes = [(batch_size, s, s) for s in tta_sizes]
    shapes.append((batch_size, cfg.input_height, cfg.input_width))

    with logger.phase(f'warmup_{strategy_name}'):
        wt_start = time.time()
        warmup_model(model, device, shapes, forward_fn=forward_fn,
                     num_steps=1, use_amp=use_amp)
        warmup_time = time.time() - wt_start

    input_shape = (batch_size, cfg.input_height, cfg.input_width)
    perf = measure_compile_performance(model, device, input_shape,
                                       num_warmup=3, num_iters=20,
                                       use_amp=use_amp, forward_fn=forward_fn)
    result = {
        'strategy': strategy_name,
        'success': True,
        'compile_time_s': compile_time,
        'warmup_time_s': warmup_time,
        'total_setup_time_s': compile_time + warmup_time,
        'num_tta_shapes': len(shapes),
        **perf,
    }
    logger.log_strategy_result(strategy_name, compile_time, warmup_time,
                               compile_time + warmup_time, perf)
    return model, result


def warmup_strategy_fixed_shape(model, device, mode='default',
                                input_shape=None, use_amp=True, forward_fn=None):
    logger = CompileLogger.get_instance()
    strategy_name = 'fixed_shape_no_dynamic'

    if input_shape is None:
        from config import config as cfg
        input_shape = (cfg.batch_size, cfg.input_height, cfg.input_width)

    compile_start = time.time()
    model, ok = try_compile_model(model, mode=mode, dynamic=False)
    compile_time = time.time() - compile_start
    if not ok:
        return model, {'strategy': strategy_name, 'success': False}

    with logger.phase(f'warmup_{strategy_name}'):
        wt_start = time.time()
        warmup_model(model, device, [input_shape], forward_fn=forward_fn,
                     num_steps=2, use_amp=use_amp)
        warmup_time = time.time() - wt_start

    perf = measure_compile_performance(model, device, input_shape,
                                       num_warmup=3, num_iters=20,
                                       use_amp=use_amp, forward_fn=forward_fn)
    result = {
        'strategy': strategy_name,
        'success': True,
        'compile_time_s': compile_time,
        'warmup_time_s': warmup_time,
        'total_setup_time_s': compile_time + warmup_time,
        **perf,
    }
    logger.log_strategy_result(strategy_name, compile_time, warmup_time,
                               compile_time + warmup_time, perf)
    return model, result


def warmup_strategy_cache_optimized(model, device, mode='default',
                                    dynamic=True, input_shape=None,
                                    use_amp=True, forward_fn=None,
                                    cache_dir=None):
    logger = CompileLogger.get_instance()
    strategy_name = 'cache_optimized'

    configure_compile_cache(cache_dir)
    configure_dynamo_cache(cache_size_limit=512, accumulated_cache_size_limit=1024)

    if input_shape is None:
        from config import config as cfg
        input_shape = (cfg.batch_size, cfg.input_height, cfg.input_width)

    compile_start = time.time()
    model, ok = try_compile_model(model, mode=mode, dynamic=dynamic)
    compile_time = time.time() - compile_start
    if not ok:
        return model, {'strategy': strategy_name, 'success': False}

    with logger.phase(f'warmup_{strategy_name}'):
        wt_start = time.time()
        warmup_model(model, device, [input_shape], forward_fn=forward_fn,
                     num_steps=2, use_amp=use_amp)
        warmup_time = time.time() - wt_start

    perf = measure_compile_performance(model, device, input_shape,
                                       num_warmup=3, num_iters=20,
                                       use_amp=use_amp, forward_fn=forward_fn)
    result = {
        'strategy': strategy_name,
        'success': True,
        'compile_time_s': compile_time,
        'warmup_time_s': warmup_time,
        'total_setup_time_s': compile_time + warmup_time,
        **perf,
    }
    logger.log_strategy_result(strategy_name, compile_time, warmup_time,
                               compile_time + warmup_time, perf)
    return model, result


def warmup_strategy_selective(model, device, mode='default',
                              compile_submodules=None, input_shape=None,
                              use_amp=True, forward_fn=None):
    logger = CompileLogger.get_instance()
    strategy_name = 'selective_compile'

    if compile_submodules is None:
        compile_submodules = ['backbone']
    if input_shape is None:
        from config import config as cfg
        input_shape = (cfg.batch_size, cfg.input_height, cfg.input_width)

    compile_start = time.time()
    model, ok = selective_compile(model, mode=mode,
                                  compile_submodules=compile_submodules)
    compile_time = time.time() - compile_start
    if not ok:
        return model, {'strategy': strategy_name, 'success': False}

    with logger.phase(f'warmup_{strategy_name}'):
        wt_start = time.time()
        warmup_model(model, device, [input_shape], forward_fn=forward_fn,
                     num_steps=2, use_amp=use_amp)
        warmup_time = time.time() - wt_start

    perf = measure_compile_performance(model, device, input_shape,
                                       num_warmup=3, num_iters=20,
                                       use_amp=use_amp, forward_fn=forward_fn)
    result = {
        'strategy': strategy_name,
        'success': True,
        'compile_time_s': compile_time,
        'warmup_time_s': warmup_time,
        'total_setup_time_s': compile_time + warmup_time,
        'compiled_submodules': compile_submodules,
        **perf,
    }
    logger.log_strategy_result(strategy_name, compile_time, warmup_time,
                               compile_time + warmup_time, perf)
    return model, result


def warmup_strategy_max_autotune(model, device, input_shape=None,
                                 use_amp=True, forward_fn=None):
    logger = CompileLogger.get_instance()
    strategy_name = 'max_autotune'

    if input_shape is None:
        from config import config as cfg
        input_shape = (cfg.batch_size, cfg.input_height, cfg.input_width)

    compile_start = time.time()
    model, ok = try_compile_model(model, mode='max-autotune', dynamic=True)
    compile_time = time.time() - compile_start
    if not ok:
        return model, {'strategy': strategy_name, 'success': False}

    with logger.phase(f'warmup_{strategy_name}'):
        wt_start = time.time()
        warmup_model(model, device, [input_shape], forward_fn=forward_fn,
                     num_steps=3, use_amp=use_amp)
        warmup_time = time.time() - wt_start

    perf = measure_compile_performance(model, device, input_shape,
                                       num_warmup=3, num_iters=20,
                                       use_amp=use_amp, forward_fn=forward_fn)
    result = {
        'strategy': strategy_name,
        'success': True,
        'compile_time_s': compile_time,
        'warmup_time_s': warmup_time,
        'total_setup_time_s': compile_time + warmup_time,
        **perf,
    }
    logger.log_strategy_result(strategy_name, compile_time, warmup_time,
                               compile_time + warmup_time, perf)
    return model, result


WARMUP_STRATEGIES = {
    'single_shape': warmup_strategy_single,
    'multi_shape_tta': warmup_strategy_tta,
    'fixed_shape': warmup_strategy_fixed_shape,
    'cache_optimized': warmup_strategy_cache_optimized,
    'selective': warmup_strategy_selective,
    'max_autotune': warmup_strategy_max_autotune,
}


def _estimate_model_vram_gb(model, device):
    if not t.cuda.is_available():
        return 0
    try:
        param_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
        props = t.cuda.get_device_properties(0)
        total_vram = getattr(props, 'total_mem', getattr(props, 'total_memory', 0)) / (1024**3)
        return param_mem, total_vram
    except Exception:
        return 0, 0


def _safe_input_shape_for_vram(input_shape, model_param_gb, total_vram_gb, headroom=0.2):
    bs, h, w = input_shape
    avail_gb = total_vram_gb * (1 - headroom) - model_param_gb * 2.5
    if avail_gb <= 0:
        return (1, h, w)
    per_sample_mb = 3 * h * w * 4 / (1024**2)
    per_batch_mb = per_sample_mb * bs
    needed_gb = per_batch_mb / 1024 * 4
    if needed_gb <= avail_gb:
        return input_shape
    safe_bs = max(1, int(avail_gb * 1024 / (per_batch_mb / bs * 4)))
    return (safe_bs, h, w)


def run_compile_audit(model, device, strategies=None, input_shape=None,
                      use_amp=True, forward_fn=None, oom_safe=True):
    import gc
    import copy
    logger = CompileLogger.get_instance()
    logger.logger.info('=' * 60)
    logger.logger.info('[AUDIT] Starting compile audit')
    logger.logger.info('=' * 60)

    if strategies is None:
        strategies = list(WARMUP_STRATEGIES.keys())
    if input_shape is None:
        from config import config as cfg
        input_shape = (cfg.batch_size, cfg.input_height, cfg.input_width)

    param_gb, total_vram = _estimate_model_vram_gb(model, device)
    logger.log_compile_config({
        'audit_model_param_gb': round(param_gb, 3),
        'audit_gpu_total_vram_gb': round(total_vram, 1),
        'audit_oom_safe': oom_safe,
        'audit_input_shape': list(input_shape),
    })

    if oom_safe and total_vram > 0:
        safe_shape = _safe_input_shape_for_vram(input_shape, param_gb, total_vram)
        if safe_shape[0] != input_shape[0]:
            logger.logger.info(
                f'[AUDIT] OOM-safe: reducing batch_size {input_shape[0]} -> {safe_shape[0]} '
                f'(param={param_gb:.2f}GB, vram={total_vram:.1f}GB)')
            input_shape = safe_shape

    all_results = []
    oom_hit_strategies = []

    for strategy_name in strategies:
        if strategy_name not in WARMUP_STRATEGIES:
            logger.logger.warning(f'[AUDIT] Unknown strategy: {strategy_name}, skipping')
            continue

        logger.logger.info(f'\n[AUDIT] Testing strategy: {strategy_name}')
        gc.collect()
        t.cuda.empty_cache()
        if t.cuda.is_available():
            t.cuda.reset_peak_memory_stats()

        audit_model = copy.deepcopy(model).to(device)
        if hasattr(audit_model, 'train'):
            audit_model.eval()

        strategy_fn = WARMUP_STRATEGIES[strategy_name]
        try:
            audit_model, result = strategy_fn(
                audit_model, device, input_shape=input_shape,
                use_amp=use_amp, forward_fn=forward_fn)
            all_results.append(result)
            logger.logger.info(
                f'[AUDIT] {strategy_name}: setup={result.get("total_setup_time_s", 0):.1f}s '
                f'avg_latency={result.get("avg_latency_ms", 0):.2f}ms')
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                logger.logger.warning(f'[AUDIT] {strategy_name} hit OOM: {e}')
                oom_hit_strategies.append(strategy_name)
                all_results.append({
                    'strategy': strategy_name,
                    'success': False,
                    'oom': True,
                    'error': str(e),
                })
                t.cuda.empty_cache()

                reduced_bs = max(1, input_shape[0] // 4)
                if reduced_bs < input_shape[0]:
                    reduced_shape = (reduced_bs, input_shape[1], input_shape[2])
                    logger.logger.info(
                        f'[AUDIT] Retrying {strategy_name} with bs={reduced_bs}')
                    del audit_model
                    gc.collect()
                    t.cuda.empty_cache()

                    audit_model = copy.deepcopy(model).to(device)
                    if hasattr(audit_model, 'train'):
                        audit_model.eval()
                    try:
                        audit_model, result = strategy_fn(
                            audit_model, device, input_shape=reduced_shape,
                            use_amp=use_amp, forward_fn=forward_fn)
                        result['oom_retry'] = True
                        result['reduced_batch_size'] = reduced_bs
                        all_results[-1] = result
                        logger.logger.info(
                            f'[AUDIT] {strategy_name} retry OK: '
                            f'setup={result.get("total_setup_time_s", 0):.1f}s '
                            f'avg_latency={result.get("avg_latency_ms", 0):.2f}ms '
                            f'(bs={reduced_bs})')
                    except Exception as e2:
                        logger.logger.warning(f'[AUDIT] {strategy_name} retry also failed: {e2}')
                        all_results[-1] = {
                            'strategy': strategy_name,
                            'success': False,
                            'oom': True,
                            'retry_failed': True,
                            'error': str(e2),
                        }
            else:
                logger.logger.info(f'[AUDIT] {strategy_name} failed: {e}')
                all_results.append({
                    'strategy': strategy_name,
                    'success': False,
                    'error': str(e),
                })
        except Exception as e:
            logger.logger.info(f'[AUDIT] {strategy_name} failed: {e}')
            all_results.append({
                'strategy': strategy_name,
                'success': False,
                'error': str(e),
            })

        del audit_model
        gc.collect()
        t.cuda.empty_cache()

    if oom_hit_strategies:
        logger.logger.warning(
            f'[AUDIT] OOM strategies: {oom_hit_strategies}. '
            f'Consider: reduce batch_size, use selective compile, or disable fullgraph.')

    summary = logger.print_summary()
    summary['oom_hit_strategies'] = oom_hit_strategies
    summary['audit_results'] = all_results
    return all_results, summary
