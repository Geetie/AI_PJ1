import os
import sys
import unittest
import time
import json
import tempfile
import shutil

import torch as t
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config, GPUProfile, CPUProfile, AMDLargeProfile, A100Profile
from utils.compile_utils import (
    is_compile_available, is_triton_available, get_raw_model,
    try_compile_model, configure_dynamo_cache, configure_compile_cache,
    CompileLogger, selective_compile, measure_compile_performance,
    WARMUP_STRATEGIES, _estimate_model_vram_gb, _safe_input_shape_for_vram
)
from models import create_model
from models.multihead import DigitsResnet101, PositionAwareAttentionHead, CrossHeadCommLayer, HeadInteractionLayer
from models.ctc import CTCModel
from models.backbone import FPNBackbone, SEBlock
from models.transformer import TransformerDigitsModel
from losses.classification import LabelSmoothEntropy
from losses.attention import AttentionSupervisionLoss, attention_diversity_loss, spatial_ordering_loss
from losses.augmentation import cutmix_data
from inference.decode import parse2class, parse2class_from_probs, ctc_greedy_decode, ctc_beam_decode
from utils.misc import PadToSquare, write2csv, find_latest_checkpoint
from utils.seed import set_seed


class TestConfig(unittest.TestCase):
    def test_config_has_required_attrs(self):
        required = [
            'batch_size', 'lr', 'class_num', 'num_heads', 'input_height', 'input_width',
            'epoches', 'warmup_epochs', 'dropout', 'fc_hidden', 'ema_decay',
            'use_torch_compile', 'compile_mode', 'compile_dynamic', 'compile_fullgraph',
            'grad_accum_steps', 'eval_batch_size', 'tta_sizes', 'smooth', 'erase_prob',
            'bbox_loss_weight', 'attn_diversity_weight', 'attn_supervision_weight',
            'ordering_loss_weight', 'aux_loss_weight', 'use_gradient_checkpoint',
            'num_workers', 'pin_memory', 'prefetch_factor', 'persistent_workers',
            'multiprocessing_context', 'oom_headroom_ratio', 'max_checkpoints',
            'multiscale_feat_dim', 'pos_embed_channels', 'feat_spatial_size',
            'roi_feat_dim', 'transformer_heads', 'transformer_layers',
            'head_interaction_layers', 'num_attn_channels', 'cutmix_alpha',
            'cutmix_prob', 'aug_rotation_degrees', 'aug_blur_prob', 'roi_gt_prob',
            'backbone_lr_factor', 'model_type', 'keep_aspect_ratio', 'roi_refine',
            'roi_teacher_forcing', 'early_stopping_patience', 'resize_size',
        ]
        for attr in required:
            self.assertTrue(hasattr(config, attr), f'Config missing attr: {attr}')

    def test_gpu_profiles_inherit(self):
        self.assertTrue(issubclass(CPUProfile, GPUProfile))
        self.assertTrue(issubclass(AMDLargeProfile, GPUProfile))
        self.assertTrue(issubclass(A100Profile, GPUProfile))

    def test_amd_large_profile_values(self):
        p = AMDLargeProfile()
        self.assertEqual(p.input_height, 416)
        self.assertEqual(p.input_width, 416)
        self.assertTrue(p.use_torch_compile)
        self.assertFalse(p.use_gradient_checkpoint)
        self.assertEqual(p.grad_accum_steps, 2)
        self.assertFalse(p.pin_memory)
        self.assertEqual(p.num_workers, 16)

    def test_config_class_num(self):
        self.assertEqual(config.class_num, 11)
        self.assertEqual(config.num_heads, 6)

    def test_data_dirs_exist_as_strings(self):
        from config import data_dir
        for key in ['train_data', 'val_data', 'test_data', 'train_label', 'val_label', 'submit_file']:
            self.assertIn(key, data_dir)
            self.assertIsInstance(data_dir[key], str)


class TestCompileUtils(unittest.TestCase):
    def test_is_triton_available_returns_bool(self):
        result = is_triton_available()
        self.assertIsInstance(result, bool)

    def test_is_compile_available_returns_bool(self):
        result = is_compile_available()
        self.assertIsInstance(result, bool)

    def test_get_raw_model_no_compile(self):
        model = nn.Linear(10, 5)
        raw = get_raw_model(model)
        self.assertIs(raw, model)

    def test_get_raw_model_with_compile_wrapper(self):
        model = nn.Linear(10, 5)
        model._orig_mod = nn.Linear(10, 5)
        raw = get_raw_model(model)
        self.assertIs(raw, model._orig_mod)

    def test_get_raw_model_nested(self):
        model = nn.Linear(10, 5)
        inner = nn.Linear(10, 5)
        inner._orig_mod = nn.Linear(10, 5)
        model._orig_mod = inner
        raw = get_raw_model(model)
        self.assertIs(raw, inner._orig_mod)

    def test_configure_dynamo_cache(self):
        configure_dynamo_cache(cache_size_limit=128, accumulated_cache_size_limit=256)

    def test_configure_compile_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = configure_compile_cache(cache_dir=tmpdir)
            self.assertTrue(os.path.exists(cache_dir))

    def test_compile_logger_singleton(self):
        CompileLogger.reset_instance()
        l1 = CompileLogger.get_instance()
        l2 = CompileLogger.get_instance()
        self.assertIs(l1, l2)
        CompileLogger.reset_instance()

    def test_compile_logger_events(self):
        CompileLogger.reset_instance()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            logger = CompileLogger(log_dir=tmpdir)
            logger.log_compile_start('test_model', mode='default')
            logger.log_compile_end(success=True, compile_time_s=1.0)
            logger.log_warmup_shape((32, 384, 384), 0, 0.5, is_first_compile=True)
            logger.log_warmup_summary(1.0, 1, 1)
            logger.log_inference_perf((32, 384, 384), 0.5, 10.0, 10.0, 12.0, 9.0, 11.0)
            logger.log_phase_start('test_phase')
            logger.log_phase_end('test_phase')
            logger.log_compile_config({'test_key': 'test_val'})
            summary = logger.get_summary()
            self.assertEqual(summary['total_compile_attempts'], 1)
            self.assertEqual(summary['successful_compiles'], 1)
            for h in logger.logger.handlers:
                h.close()
            logger.save_json_log()
        CompileLogger.reset_instance()

    def test_selective_compile_no_compile_available(self):
        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 3))
        result_model, ok = selective_compile(model, compile_submodules=['0'])
        if not is_compile_available():
            self.assertFalse(ok)

    def test_estimate_model_vram(self):
        model = nn.Linear(1000, 1000)
        result = _estimate_model_vram_gb(model, t.device('cpu'))
        if t.cuda.is_available():
            param_gb, total_vram = result
            self.assertGreater(total_vram, 0)
        else:
            self.assertEqual(result, 0)

    def test_safe_input_shape_for_vram(self):
        result = _safe_input_shape_for_vram((256, 416, 416), 0.5, 192.0, headroom=0.2)
        self.assertEqual(len(result), 3)

    def test_warmup_strategies_dict(self):
        expected = ['single_shape', 'multi_shape_tta', 'fixed_shape', 'cache_optimized', 'selective', 'max_autotune']
        for name in expected:
            self.assertIn(name, WARMUP_STRATEGIES)


class TestModels(unittest.TestCase):
    def test_create_model_fpn_multihead(self):
        model = create_model('fpn_multihead')
        self.assertIsInstance(model, DigitsResnet101)

    def test_create_model_ctc(self):
        model = create_model('ctc')
        self.assertIsInstance(model, CTCModel)

    def test_create_model_transformer(self):
        model = create_model('transformer')
        self.assertIsInstance(model, TransformerDigitsModel)

    def test_create_model_default(self):
        model = create_model()
        self.assertIsInstance(model, DigitsResnet101)

    def test_fpn_backbone_output_shape(self):
        backbone = FPNBackbone()
        backbone.eval()
        x = t.randn(2, 3, config.input_height, config.input_width)
        with t.no_grad():
            out = backbone(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], config.multiscale_feat_dim)

    def test_digits_resnet101_forward(self):
        model = DigitsResnet101(config.class_num, config.num_heads)
        model.eval()
        x = t.randn(2, 3, config.input_height, config.input_width)
        with t.no_grad():
            cls_outs, bbox_outs = model(x)
        self.assertEqual(len(cls_outs), config.num_heads)
        self.assertEqual(len(bbox_outs), config.num_heads)
        for h in range(config.num_heads):
            self.assertEqual(cls_outs[h].shape, (2, config.class_num))
            self.assertEqual(bbox_outs[h].shape, (2, 4))

    def test_digits_resnet101_forward_with_attn(self):
        model = DigitsResnet101(config.class_num, config.num_heads)
        model.eval()
        x = t.randn(2, 3, config.input_height, config.input_width)
        with t.no_grad():
            cls_list, bbox_tuple, attn_maps, head_cls_outs = model.forward_with_attn(x)
        self.assertEqual(len(cls_list), config.num_heads)
        self.assertEqual(len(attn_maps), config.num_heads)
        self.assertEqual(len(head_cls_outs), config.num_heads)

    def test_digits_resnet101_forward_with_probs(self):
        model = DigitsResnet101(config.class_num, config.num_heads)
        model.eval()
        x = t.randn(2, 3, config.input_height, config.input_width)
        with t.no_grad():
            probs = model.forward_with_probs(x)
        self.assertEqual(len(probs), config.num_heads)
        for p in probs:
            self.assertTrue(torch_close(p.sum(dim=1), t.ones(2), atol=1e-4))

    def test_ctc_model_output_shape(self):
        model = CTCModel(num_classes=config.class_num)
        model.eval()
        x = t.randn(4, 3, config.input_height, config.input_width)
        with t.no_grad():
            out = model(x)
        self.assertEqual(len(out.shape), 3)
        self.assertEqual(out.shape[1], 4)
        self.assertEqual(out.shape[2], config.class_num)

    def test_transformer_model_forward(self):
        model = TransformerDigitsModel(config.class_num, config.num_heads)
        model.eval()
        x = t.randn(2, 3, config.input_height, config.input_width)
        with t.no_grad():
            cls_outs, bbox_outs = model(x)
        self.assertEqual(len(cls_outs), config.num_heads)
        self.assertEqual(len(bbox_outs), config.num_heads)

    def test_se_block(self):
        se = SEBlock(64)
        x = t.randn(2, 64, 10, 10)
        out = se(x)
        self.assertEqual(out.shape, x.shape)

    def test_position_aware_attention_head(self):
        head = PositionAwareAttentionHead(config.multiscale_feat_dim, config.fc_hidden, config.class_num, head_idx=0)
        head.eval()
        x = t.randn(2, config.multiscale_feat_dim, config.feat_spatial_size, config.feat_spatial_size)
        with t.no_grad():
            cls_out, bbox_out, hidden = head(x, return_attn=False)
        self.assertEqual(cls_out.shape, (2, config.class_num))
        self.assertEqual(bbox_out.shape, (2, 4))

    def test_head_interaction_layer(self):
        layer = HeadInteractionLayer(config.fc_hidden, config.num_heads, num_layers=2, nhead=4)
        head_feats = [t.randn(2, config.fc_hidden) for _ in range(config.num_heads)]
        out = layer(head_feats)
        self.assertEqual(len(out), config.num_heads)
        for f in out:
            self.assertEqual(f.shape, (2, config.fc_hidden))

    def test_set_roi_gt_prob(self):
        model = DigitsResnet101(config.class_num, config.num_heads)
        model.set_roi_gt_prob(0.5)
        self.assertEqual(model.roi_gt_prob, 0.5)

    def test_model_train_eval_mode(self):
        model = DigitsResnet101(config.class_num, config.num_heads)
        model.train()
        self.assertTrue(model.training)
        model.eval()
        self.assertFalse(model.training)


class TestLosses(unittest.TestCase):
    def test_label_smooth_entropy(self):
        criterion = LabelSmoothEntropy(smooth=0.1)
        preds = t.randn(8, 11)
        targets = t.randint(0, 11, (8,))
        loss = criterion(preds, targets)
        self.assertEqual(loss.shape, ())
        self.assertTrue(loss.item() >= 0)

    def test_label_smooth_entropy_with_weights(self):
        weights = t.ones(11)
        criterion = LabelSmoothEntropy(smooth=0.1, class_weights=weights)
        preds = t.randn(8, 11)
        targets = t.randint(0, 11, (8,))
        loss = criterion(preds, targets)
        self.assertEqual(loss.shape, ())

    def test_attention_supervision_loss(self):
        loss_fn = AttentionSupervisionLoss()
        attn_maps = [t.randn(4, 1, 20, 20).softmax(dim=2).softmax(dim=3) for _ in range(3)]
        bbox_target = t.rand(4, 3, 4)
        bbox_mask = t.ones(4, 3)
        bbox_mask[:, 2] = 0
        loss = loss_fn(attn_maps, bbox_target, bbox_mask)
        self.assertEqual(loss.shape, ())

    def test_attention_supervision_loss_empty(self):
        loss_fn = AttentionSupervisionLoss()
        loss = loss_fn([], t.rand(4, 3, 4), t.ones(4, 3))
        self.assertEqual(loss.shape, ())

    def test_attention_diversity_loss(self):
        attn_maps = [t.randn(4, 1, 20, 20).softmax(dim=2).softmax(dim=3) for _ in range(3)]
        loss = attention_diversity_loss(attn_maps)
        self.assertEqual(loss.shape, ())

    def test_spatial_ordering_loss(self):
        attn_maps = [t.randn(4, 1, 20, 20).softmax(dim=2).softmax(dim=3) for _ in range(3)]
        loss = spatial_ordering_loss(attn_maps)
        self.assertEqual(loss.shape, ())

    def test_cutmix_data(self):
        x = t.randn(8, 3, 224, 224)
        y = t.randint(0, 11, (8, 6))
        bbox_target = t.rand(8, 6, 4)
        bbox_mask = t.ones(8, 6)
        result = cutmix_data(x, y, bbox_target, bbox_mask, alpha=1.0)
        self.assertEqual(len(result), 8)
        mixed_x, y_a, y_b, bbox_a, bbox_b, mask_a, mask_b, lam = result
        self.assertEqual(mixed_x.shape, x.shape)
        self.assertTrue(0 <= lam <= 1)


class TestDecode(unittest.TestCase):
    def test_parse2class(self):
        pred = [t.randn(4, 11) for _ in range(6)]
        pred[0][0, 3] = 100
        pred[1][0, 5] = 100
        pred[2][0, 10] = 100
        result = parse2class(pred)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], '35')

    def test_parse2class_from_probs(self):
        probs = [t.zeros(4, 11) for _ in range(6)]
        probs[0][0, 3] = 1.0
        probs[1][0, 5] = 1.0
        probs[2][0, 10] = 1.0
        result = parse2class_from_probs(probs)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], '35')

    def test_ctc_greedy_decode(self):
        T, B, C = 10, 2, 11
        log_probs = t.zeros(T, B, C)
        log_probs[:, :, 10] = -0.1
        log_probs[2, 0, 3] = 5.0
        log_probs[4, 0, 5] = 5.0
        log_probs[3, 1, 7] = 5.0
        result = ctc_greedy_decode(log_probs)
        self.assertEqual(len(result), 2)
        self.assertIn(3, result[0])
        self.assertIn(5, result[0])
        self.assertIn(7, result[1])

    def test_ctc_beam_decode(self):
        T, B, C = 10, 2, 11
        log_probs = t.zeros(T, B, C)
        log_probs[:, :, 10] = -0.1
        log_probs[2, 0, 3] = 5.0
        log_probs[4, 0, 5] = 5.0
        result = ctc_beam_decode(log_probs, beam_width=5)
        self.assertEqual(len(result), 2)


class TestMisc(unittest.TestCase):
    def test_pad_to_square(self):
        pad = PadToSquare(fill=127)
        from PIL import Image
        img = Image.new('RGB', (100, 200), (0, 0, 0))
        padded = pad(img)
        self.assertGreaterEqual(padded.size[0], 100)
        self.assertGreaterEqual(padded.size[1], 200)
        self.assertEqual(max(padded.size), padded.size[0])
        self.assertEqual(max(padded.size), padded.size[1])

    def test_write2csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'test.csv')
            results = [['img1.png', '123'], ['img2.png', '456']]
            write2csv(results, csv_path)
            self.assertTrue(os.path.exists(csv_path))

    def test_set_seed(self):
        set_seed(42)
        a = t.randn(10)
        set_seed(42)
        b = t.randn(10)
        self.assertTrue(t.allclose(a, b))


class TestEMABuffers(unittest.TestCase):
    def test_ema_updates_buffers(self):
        from trainer.base import ModelEMA
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 10),
        )
        x = t.randn(4, 3, 4, 4)
        model.train()
        _ = model(x)

        ema = ModelEMA(model, decay=0.9)

        bn_running_mean_before = ema.ema[1].running_mean.clone()
        model.train()
        _ = model(x + 1.0)
        ema.update(model)
        bn_running_mean_after = ema.ema[1].running_mean.clone()

        self.assertFalse(t.allclose(bn_running_mean_before, bn_running_mean_after),
                        "EMA buffers should be updated after update()")

    def test_ema_to_device(self):
        from trainer.base import ModelEMA
        model = nn.Linear(10, 5)
        ema = ModelEMA(model, decay=0.999)
        result = ema.to_device(t.device('cpu'))
        self.assertIsInstance(result, nn.Module)


class TestGradAccumConsistency(unittest.TestCase):
    def test_multihead_trainer_has_grad_accum(self):
        from trainer.multihead import MultiHeadTrainer
        self.assertTrue(hasattr(MultiHeadTrainer, '_train_epoch'))

    def test_ctc_trainer_has_grad_accum(self):
        from trainer.ctc import CTCTrainer
        self.assertTrue(hasattr(CTCTrainer, '_train_epoch'))

    def test_ctc_eval_uses_to_device(self):
        import inspect
        from trainer.ctc import CTCTrainer
        source = inspect.getsource(CTCTrainer._eval)
        self.assertIn('to_device', source)
        self.assertNotIn('self.ema.ema', source)


class TestGradientCheckpointCompileCompat(unittest.TestCase):
    def test_backbone_forward_uses_compile_flag(self):
        import inspect
        source = inspect.getsource(FPNBackbone.forward)
        self.assertIn('use_torch_compile', source)

    def test_multihead_forward_uses_compile_flag(self):
        import inspect
        source = inspect.getsource(DigitsResnet101.forward)
        self.assertIn('use_torch_compile', source)

    def test_multihead_forward_with_attn_uses_compile_flag(self):
        import inspect
        source = inspect.getsource(DigitsResnet101.forward_with_attn)
        self.assertIn('use_torch_compile', source)


class TestAuxLossGradient(unittest.TestCase):
    def test_head_cls_outs_not_detached(self):
        import inspect
        source = inspect.getsource(DigitsResnet101.forward_with_attn)
        self.assertNotIn('cls_out.detach()', source)


class TestOOMSchedulerFix(unittest.TestCase):
    def test_base_trainer_lr_scheduler_step_on_oom(self):
        import inspect
        from trainer.base import BaseTrainer
        source = inspect.getsource(BaseTrainer.train)
        oom_continue_count = source.count('self.lr_scheduler.step()')
        self.assertGreaterEqual(oom_continue_count, 3,
                                "lr_scheduler.step() should be called in OOM and SHM recovery paths too")


class TestGradAccumCeil(unittest.TestCase):
    def test_grad_accum_uses_ceil(self):
        import inspect
        from trainer.base import BaseTrainer
        source = inspect.getsource(BaseTrainer.train)
        self.assertIn('-(-', source)


class TestSaveModelIncludesTrainWeights(unittest.TestCase):
    def test_save_model_includes_train_model(self):
        import inspect
        from trainer.base import BaseTrainer
        source = inspect.getsource(BaseTrainer.save_model)
        self.assertIn('train_model', source)


class TestLoadModelSchedulerState(unittest.TestCase):
    def test_load_model_prefers_scheduler_state(self):
        import inspect
        from trainer.base import BaseTrainer
        source = inspect.getsource(BaseTrainer.load_model)
        self.assertIn("if 'lr_scheduler' in dicts", source)


class TestMakeLoaderInterface(unittest.TestCase):
    def test_multihead_make_loader_has_collate_fn(self):
        import inspect
        from trainer.multihead import MultiHeadTrainer
        source = inspect.getsource(MultiHeadTrainer._make_loader)
        self.assertIn('collate_fn', source)


class TestClassWeightExcludesEmpty(unittest.TestCase):
    def test_compute_class_weights_excludes_class10(self):
        import inspect
        from trainer.multihead import MultiHeadTrainer
        source = inspect.getsource(MultiHeadTrainer._compute_class_weights)
        self.assertIn('class_weights[10] = 0.0', source)


class TestBeamDecodePrealloc(unittest.TestCase):
    def test_beam_decode_returns_correct_length(self):
        T, B, C = 5, 3, 11
        log_probs = t.zeros(T, B, C)
        log_probs[:, :, 10] = -0.1
        result = ctc_beam_decode(log_probs, beam_width=3)
        self.assertEqual(len(result), B)


class TestCrossEnsembleBatchCheck(unittest.TestCase):
    def test_ensemble_has_batch_size_check(self):
        import inspect
        from inference.predict import cross_model_ensemble
        source = inspect.getsource(cross_model_ensemble)
        self.assertIn('mismatch', source.lower())


class TestConfidenceNormalization(unittest.TestCase):
    def test_ensemble_uses_normalized_confidence(self):
        import inspect
        from inference.predict import cross_model_ensemble
        source = inspect.getsource(cross_model_ensemble)
        self.assertIn('norm_mh', source)
        self.assertIn('norm_ctc', source)


class TestWarmupDynamoReset(unittest.TestCase):
    def test_warmup_resets_dynamo(self):
        import inspect
        from trainer.multihead import MultiHeadTrainer
        source = inspect.getsource(MultiHeadTrainer._gpu_warmup)
        self.assertIn('_dynamo.reset', source)


class TestCompileFallbackRebuildsEMA(unittest.TestCase):
    def test_compile_fallback_rebuilds_ema(self):
        import inspect
        from trainer.multihead import MultiHeadTrainer
        source = inspect.getsource(MultiHeadTrainer._gpu_warmup)
        self.assertIn('ModelEMA', source.split('use_torch_compile = False')[1])


class TestForkCUDACheck(unittest.TestCase):
    def test_config_checks_cuda_before_fork(self):
        import inspect
        from config import _detect_gpu_profile
        source = inspect.getsource(_detect_gpu_profile)
        self.assertIn('cuda.is_initialized', source)


def torch_close(a, b, atol=1e-5):
    return t.allclose(a, b, atol=atol)


if __name__ == '__main__':
    unittest.main(verbosity=2)
