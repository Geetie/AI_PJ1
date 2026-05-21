#!/usr/bin/env python3
"""
优化训练脚本 - 解决OOM问题的启动脚本

设置了以下优化环境变量：
1. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True - 减少显存碎片
2. CUDA_LAUNCH_BLOCKING=0 - 保持异步以获得最佳性能
"""

import os
import sys

# 设置优化环境变量
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    from trainer.multihead import MultiHeadTrainer
    
    print('=' * 60)
    print('🎯 优化训练脚本启动')
    print('=' * 60)
    print(f"环境变量 PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
    print(f"环境变量 CUDA_LAUNCH_BLOCKING: {os.environ.get('CUDA_LAUNCH_BLOCKING')}")
    print('=' * 60)
    
    try:
        print('无 checkpoint，从头训练')
        trainer = MultiHeadTrainer(model_type='fpn_multihead')
        trainer.train()
    except Exception as e:
        print(f'训练过程中发生错误: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
