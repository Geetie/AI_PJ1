import os
import sys
import multiprocessing
import torch as t

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
BASE_DIR = '/mnt/workspace' if os.path.exists('/mnt/workspace') else PROJECT_DIR


def is_amd_rocm():
    return hasattr(t.version, 'hip') and t.version.hip is not None


def is_nvidia_cuda():
    if not t.cuda.is_available():
        return False
    if is_amd_rocm():
        return False
    try:
        props = t.cuda.get_device_properties(0)
        gpu_name = props.name.lower()
        if 'amd' in gpu_name or 'radeon' in gpu_name or 'instinct' in gpu_name:
            return False
        return True
    except Exception:
        return False


def is_cpu_only():
    return not t.cuda.is_available()


def get_platform():
    if is_cpu_only():
        return 'cpu'
    if is_amd_rocm():
        return 'amd_rocm'
    return 'nvidia_cuda'


GPU_PLATFORM = get_platform()


def get_total_vram_gb():
    if not t.cuda.is_available():
        return 0
    try:
        props = t.cuda.get_device_properties(0)
        return getattr(props, 'total_mem', getattr(props, 'total_memory', 0)) / (1024 ** 3)
    except Exception:
        return 0


TOTAL_VRAM_GB = get_total_vram_gb()
NUM_PHYSICAL_CORES = multiprocessing.cpu_count() or 2


def get_optimal_device():
    return t.device('cuda') if t.cuda.is_available() else t.device('cpu')


def get_device_properties():
    if not t.cuda.is_available():
        return {}
    try:
        props = t.cuda.get_device_properties(0)
        return {
            'name': props.name,
            'total_mem_gb': getattr(props, 'total_mem', getattr(props, 'total_memory', 0)) / (1024 ** 3),
            'compute_capability': getattr(props, 'major', 0) * 10 + getattr(props, 'minor', 0),
            'multi_processor_count': getattr(props, 'multi_processor_count', 0),
            'platform': GPU_PLATFORM,
        }
    except Exception:
        return {}


def get_precision_config():
    config = {
        'tf32_enabled': False,
        'allow_tf32': False,
        'matmul_precision': 'highest',
    }

    if is_nvidia_cuda():
        try:
            t.set_float32_matmul_precision('high')
            config['matmul_precision'] = 'high'
            config['tf32_enabled'] = True
        except Exception:
            pass
        try:
            t.backends.cudnn.allow_tf32 = True
            config['allow_tf32'] = True
        except Exception:
            pass

    elif is_amd_rocm():
        try:
            t.set_float32_matmul_precision('high')
            config['matmul_precision'] = 'high'
        except Exception:
            pass

    return config


def get_blas_backend():
    if is_nvidia_cuda():
        return 'cublaslt'
    if is_amd_rocm():
        try:
            current = t.backends.cuda.preferred_blas_library()
            if current == 'cublaslt':
                t.backends.cuda.preferred_blas_library('cublas')
                return 'rocblas'
        except Exception:
            pass
        return 'rocblas'
    return 'none'


def get_cache_dirs():
    dirs = {
        'inductor': os.path.join(BASE_DIR, 'inductor_cache'),
        'triton': os.path.join(BASE_DIR, 'triton_cache'),
        'miopen': os.path.join(BASE_DIR, 'miopen_cache'),
    }

    if is_amd_rocm():
        dirs['inductor'] = os.path.join(BASE_DIR, 'inductor_cache_rocm')
        dirs['triton'] = os.path.join(BASE_DIR, 'triton_cache_rocm')
    elif is_nvidia_cuda():
        dirs['inductor'] = os.path.join(BASE_DIR, 'inductor_cache_nvidia')
        dirs['triton'] = os.path.join(BASE_DIR, 'triton_cache_nvidia')

    for key, path in dirs.items():
        os.makedirs(path, exist_ok=True)

    return dirs


def get_dataloader_context():
    if GPU_PLATFORM == 'amd_rocm':
        return 'spawn'
    if GPU_PLATFORM == 'nvidia_cuda':
        return 'fork'
    return None


def get_platform_info():
    info = {
        'platform': GPU_PLATFORM,
        'vram_gb': TOTAL_VRAM_GB,
        'cpu_cores': NUM_PHYSICAL_CORES,
        'device': str(get_optimal_device()),
    }

    if is_amd_rocm():
        info.update({
            'rocm_version': getattr(t.version, 'hip', None),
            'gpu_arch': _get_gfx_arch(),
            'hsa_override': os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'auto'),
        })
    elif is_nvidia_cuda():
        info.update({
            'cuda_version': t.version.cuda,
            'gpu_name': t.cuda.get_device_name(0) if t.cuda.is_available() else 'N/A',
        })

    return info


_GFX_VERSION_MAP = {
    'gfx900': '9.0.0',
    'gfx906': '9.0.6',
    'gfx908': '9.0.8',
    'gfx90a': '9.0.10',
    'gfx940': '9.4.0',
    'gfx941': '9.4.1',
    'gfx942': '9.4.2',
    'gfx1030': '10.3.0',
    'gfx1031': '10.3.0',
    'gfx1032': '10.3.0',
    'gfx1100': '11.0.0',
    'gfx1101': '11.0.0',
    'gfx1102': '11.0.0',
    'gfx1150': '11.5.0',
    'gfx1151': '11.5.1',
    'gfx1200': '12.0.0',
    'gfx1201': '12.0.1',
}

_GPU_NAME_GFX_HINTS = {
    'mi250': 'gfx90a', 'mi210': 'gfx90a',
    'mi100': 'gfx908', 'mi50': 'gfx906',
    'mi300x': 'gfx942', 'mi300a': 'gfx941', 'mi300': 'gfx940',
    'rx 7900': 'gfx1100', 'rx 7800': 'gfx1101', 'rx 7700': 'gfx1101',
    'rx 7600': 'gfx1102',
    'rx 6900': 'gfx1030', 'rx 6800': 'gfx1030',
    'rx 6700': 'gfx1031', 'rx 6600': 'gfx1032',
}


def _get_gfx_arch():
    if not t.cuda.is_available():
        return None
    try:
        props = t.cuda.get_device_properties(0)
        gpu_name = getattr(props, 'name', '').lower()
        for hint, arch in _GPU_NAME_GFX_HINTS.items():
            if hint in gpu_name:
                return arch
        gcn_arch = getattr(props, 'gcnArchName', '').split(':')[0].strip()
        if gcn_arch and gcn_arch in _GFX_VERSION_MAP:
            return gcn_arch
        arch_list = t.cuda.get_arch_list()
        if arch_list and len(arch_list) == 1:
            return arch_list[0]
    except Exception:
        pass
    return None


def get_rocm_version():
    if not is_amd_rocm():
        return None
    try:
        parts = t.version.hip.split('.')
        major, minor = int(parts[0]), int(parts[1])
        return (major, minor)
    except Exception:
        return None


def get_gfx_arch():
    return _get_gfx_arch()


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
        try:
            import torch._inductor
            return True
        except (ImportError, OSError):
            return False
    try:
        import torch._inductor
        return True
    except (ImportError, OSError):
        return False


def is_hip_error(err_str):
    err_lower = err_str.lower()
    return any(k in err_lower for k in ['hip', 'amd', 'rocm', 'miopen', 'rocblas', 'aperture'])


def is_cuda_error(err_str):
    err_lower = err_str.lower()
    return any(k in err_lower for k in ['cuda', 'cublas', 'cudnn', 'nv'])


def is_memory_error(err_str):
    err_lower = err_str.lower()
    return any(k in err_lower for k in ['memory', 'allocation', 'oom', 'out of memory'])


def get_recommended_batch_size():
    if GPU_PLATFORM == 'amd_rocm':
        if TOTAL_VRAM_GB >= 120:
            return 256
        elif TOTAL_VRAM_GB >= 48:
            return 128
        else:
            return 64
    elif GPU_PLATFORM == 'nvidia_cuda':
        if TOTAL_VRAM_GB >= 40:
            return 128
        elif TOTAL_VRAM_GB >= 22:
            return 64
        elif TOTAL_VRAM_GB >= 16:
            return 32
        else:
            return 16
    return 16


def get_recommended_workers():
    base = max(NUM_PHYSICAL_CORES - 4, 2)
    if GPU_PLATFORM == 'amd_rocm':
        return min(base + 8, 32)
    elif GPU_PLATFORM == 'nvidia_cuda':
        return min(base, 12)
    return 2


def print_platform_info():
    info = get_platform_info()
    print("=" * 60)
    print("Platform Information")
    print("=" * 60)
    print(f"GPU Platform: {info['platform'].upper()}")
    print(f"VRAM: {info['vram_gb']:.1f} GB")
    print(f"CPU Cores: {info['cpu_cores']}")
    print(f"Device: {info['device']}")

    if GPU_PLATFORM == 'amd_rocm':
        print(f"ROCm Version: {info.get('rocm_version', 'N/A')}")
        print(f"GPU Arch: {info.get('gpu_arch', 'N/A')}")
        print(f"HSA Override: {info.get('hsa_override', 'auto')}")
        precision_config = get_precision_config()
        print(f"TF32: {'Enabled' if precision_config['tf32_enabled'] else 'N/A (AMD)'}")
    elif GPU_PLATFORM == 'nvidia_cuda':
        print(f"CUDA Version: {info.get('cuda_version', 'N/A')}")
        print(f"GPU Name: {info.get('gpu_name', 'N/A')}")
        precision_config = get_precision_config()
        print(f"TF32: {'Enabled' if precision_config['tf32_enabled'] else 'Disabled'}")

    print("=" * 60)
    return info


TRITON_AVAILABLE = is_triton_available()
COMPILE_AVAILABLE = is_compile_available()

__all__ = [
    'is_amd_rocm', 'is_nvidia_cuda', 'is_cpu_only', 'get_platform',
    'GPU_PLATFORM', 'TOTAL_VRAM_GB', 'NUM_PHYSICAL_CORES',
    'get_optimal_device', 'get_device_properties',
    'get_precision_config', 'get_blas_backend', 'get_cache_dirs',
    'get_dataloader_context', 'get_platform_info', 'get_gfx_arch',
    'is_triton_available', 'is_compile_available',
    'is_hip_error', 'is_cuda_error', 'is_memory_error',
    'get_recommended_batch_size', 'get_recommended_workers',
    'print_platform_info',
    'TRITON_AVAILABLE', 'COMPILE_AVAILABLE',
    'BASE_DIR', 'SCRIPT_DIR', 'PROJECT_DIR',
]
