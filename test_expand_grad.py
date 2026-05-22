#!/usr/bin/env python3
"""测试expand和repeat的梯度"""
import torch as t
import torch.nn as nn

# 测试1：expand
print("Test 1: expand")
x = t.randn(1, 3, 10, 10, requires_grad=True)
y = x.expand(2, 3, 10, 10)
print(f"  y.requires_grad: {y.requires_grad}")

# 测试2：repeat
print("\nTest 2: repeat")
x = t.randn(1, 3, 10, 10, requires_grad=True)
y = x.repeat(2, 1, 1, 1)
print(f"  y.requires_grad: {y.requires_grad}")

# 测试3：expand + cat + conv
print("\nTest 3: expand + cat + conv (simulate PositionAwareAttentionHead)")
class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_embed = nn.Parameter(t.randn(1, 64, 10, 10) * 0.05)
        self.conv = nn.Conv2d(3 + 64, 64, 1)

    def forward(self, x):
        pos = self.pos_embed.expand(x.shape[0], -1, -1, -1)
        out = t.cat([x, pos], dim=1)
        out = self.conv(out)
        return out

model = TestModule()
x = t.randn(2, 3, 10, 10)
y = model(x)
loss = y.sum()
loss.backward()

print(f"  pos_embed.grad: {model.pos_embed.grad.norm().item():.6f}")
print(f"  conv.weight.grad norm: {model.conv.weight.grad.norm().item():.6f}")

# 测试4：检查expand后是否在计算图中
print("\nTest 4: Check if expand is in computation graph")
x = t.randn(1, 3, 10, 10, requires_grad=True)
y = x.expand(2, 3, 10, 10)
z = y.sum()
z.backward()
print(f"  x.grad: {x.grad}")
