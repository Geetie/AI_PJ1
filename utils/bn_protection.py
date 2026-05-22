import torch as t
import torch.nn as nn
import math
import logging

logger = logging.getLogger(__name__)


class BNGradCliper:
    def __init__(self, max_norm=1.0):
        self.max_norm = max_norm
        self._clip_count = 0

    def clip(self, model):
        bn_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if 'bn' in name.lower() or 'norm' in name.lower():
                    bn_params.append(param)

        if not bn_params:
            return 0.0

        total_norm = t.norm(t.stack([p.grad.norm() for p in bn_params]), 2)
        if total_norm > self.max_norm:
            scale = self.max_norm / (total_norm + 1e-8)
            for p in bn_params:
                p.grad.mul_(scale)
            self._clip_count += 1
        return total_norm.item()


class BNMomentumScheduler:
    def __init__(self, model, warmup_epochs=10, min_momentum=0.001, max_momentum=0.1):
        self.model = model
        self.warmup_epochs = warmup_epochs
        self.min_momentum = min_momentum
        self.max_momentum = max_momentum
        self._bn_modules = []
        self._collect_bn_modules()

    def _collect_bn_modules(self):
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                self._bn_modules.append(module)

    def step(self, epoch, total_epochs=None):
        if epoch < self.warmup_epochs:
            progress = epoch / max(self.warmup_epochs, 1)
            momentum = self.min_momentum + (self.max_momentum - self.min_momentum) * progress
        else:
            if total_epochs is not None:
                remaining = total_epochs - self.warmup_epochs
                progress = min((epoch - self.warmup_epochs) / max(remaining, 1), 1.0)
                cos_factor = 0.5 * (1 + math.cos(math.pi * progress))
                momentum = self.max_momentum * cos_factor + self.min_momentum * (1 - cos_factor)
            else:
                momentum = self.max_momentum

        for m in self._bn_modules:
            m.momentum = momentum

        return momentum


class BNAnomalyDetector:
    def __init__(self, model, auto_fix=True, log_interval=100):
        self.model = model
        self.auto_fix = auto_fix
        self.log_interval = log_interval
        self._step_count = 0
        self._anomaly_history = []
        self._hooks = []

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, name):
        def hook(module, input, output):
            self._step_count += 1
            if self._step_count % self.log_interval != 0:
                return

            if not isinstance(output, t.Tensor):
                return

            has_nan = t.isnan(output).any().item()
            has_inf = t.isinf(output).any().item()
            max_val = output.abs().max().item()

            if has_nan or has_inf or max_val > 100:
                anomaly = {
                    'name': name,
                    'step': self._step_count,
                    'has_nan': has_nan,
                    'has_inf': has_inf,
                    'max_val': max_val,
                }
                self._anomaly_history.append(anomaly)
                logger.warning(
                    f'[BN-ANOMALY] {name}: nan={has_nan} inf={has_inf} '
                    f'max_val={max_val:.2f} at step {self._step_count}')

                if self.auto_fix:
                    self._fix_bn(module, name)

        return hook

    def _fix_bn(self, module, name):
        with t.no_grad():
            if isinstance(module.running_var, t.Tensor):
                bad_var = (module.running_var <= 0) | t.isnan(module.running_var) | t.isinf(module.running_var)
                if bad_var.any():
                    module.running_var[bad_var] = 1.0
                    logger.info(f'[BN-FIX] {name}: fixed {bad_var.sum().item()} bad running_var values')

            if isinstance(module.running_mean, t.Tensor):
                bad_mean = t.isnan(module.running_mean) | t.isinf(module.running_mean)
                if bad_mean.any():
                    module.running_mean[bad_mean] = 0.0
                    logger.info(f'[BN-FIX] {name}: fixed {bad_mean.sum().item()} bad running_mean values')

    def check_and_fix(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                self._fix_bn(module, name)

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def get_anomaly_report(self):
        if not self._anomaly_history:
            return "No BN anomalies detected."
        report = [f"BN Anomaly Report ({len(self._anomaly_history)} events):"]
        for a in self._anomaly_history[-20:]:
            report.append(f"  {a['name']}: nan={a['has_nan']} inf={a['has_inf']} max={a['max_val']:.2f} step={a['step']}")
        return '\n'.join(report)


class BNSafeWrapper:
    def __init__(self, model, freeze_epochs=0, eps_floor=1e-5):
        self.model = model
        self.freeze_epochs = freeze_epochs
        self.eps_floor = eps_floor
        self._bn_modules = []
        self._original_momentums = {}
        self._collect_bn_modules()

    def _collect_bn_modules(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                self._bn_modules.append((name, module))
                self._original_momentums[name] = module.momentum
                if module.eps < self.eps_floor:
                    module.eps = self.eps_floor

    def freeze_running_stats(self):
        for name, module in self._bn_modules:
            module.momentum = 0.0

    def unfreeze_running_stats(self):
        for name, module in self._bn_modules:
            module.momentum = self._original_momentums.get(name, 0.1)

    def step(self, epoch):
        if epoch < self.freeze_epochs:
            self.freeze_running_stats()
        else:
            self.unfreeze_running_stats()

    def ensure_eps_floor(self):
        for name, module in self._bn_modules:
            if module.eps < self.eps_floor:
                module.eps = self.eps_floor


def install_bn_protection(model, warmup_epochs=10, freeze_epochs=0,
                          bn_grad_max_norm=1.0, auto_fix=True):
    grad_cliper = BNGradCliper(max_norm=bn_grad_max_norm)
    momentum_scheduler = BNMomentumScheduler(model, warmup_epochs=warmup_epochs)
    anomaly_detector = BNAnomalyDetector(model, auto_fix=auto_fix)
    anomaly_detector.register_hooks()
    safe_wrapper = BNSafeWrapper(model, freeze_epochs=freeze_epochs)

    return {
        'grad_cliper': grad_cliper,
        'momentum_scheduler': momentum_scheduler,
        'anomaly_detector': anomaly_detector,
        'safe_wrapper': safe_wrapper,
    }
