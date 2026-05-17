import os
import torch as t
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import config, NUM_WORKERS, BASE_DIR
from data.dataset import DigitsDataset, CTCDataset, ctc_test_collate_fn
from models import create_model
from models.ctc import CTCModel
from inference.decode import parse2class, parse2class_from_probs, ctc_beam_decode
from utils.misc import write2csv


def predicts(model_path, csv_path, use_tta=True, model_type=None):
    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
    mt = model_type or config.model_type
    res_net = create_model(mt).to(device)
    ckpt = t.load(model_path, map_location=device, weights_only=False)
    res_net.load_state_dict(ckpt['model'])
    if 'model_type' in ckpt:
        mt = ckpt['model_type']
    print('Load model from %s successfully' % model_path)
    res_net.eval()

    if use_tta:
        test_set_for_count = DigitsDataset(mode='test', aug=False,
                                           input_size=(config.input_height, config.input_width))
        num_test = len(test_set_for_count)
        all_probs = [t.zeros(num_test, config.class_num) for _ in range(config.num_heads)]
        all_names = []

        for tta_size in config.tta_sizes:
            test_set_tta = DigitsDataset(mode='test', aug=False,
                                         input_size=(tta_size, tta_size))
            test_loader_tta = DataLoader(test_set_tta, batch_size=config.batch_size,
                                         shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
                                         drop_last=False, persistent_workers=False)
            sample_idx = 0
            with t.no_grad():
                for img, img_names in tqdm(test_loader_tta, desc=f'Test TTA size={tta_size}'):
                    img = img.to(device)
                    probs = res_net.forward_with_probs(img)
                    bs = img.size(0)
                    for h in range(config.num_heads):
                        all_probs[h][sample_idx:sample_idx + bs] += probs[h].cpu()
                    if tta_size == config.tta_sizes[0]:
                        all_names.extend(img_names)
                    sample_idx += bs

        results = [[name, code] for name, code in zip(all_names, parse2class_from_probs(all_probs))]
    else:
        test_loader = DataLoader(DigitsDataset(mode='test', aug=False,
                                               input_size=(config.input_height, config.input_width)),
                                 batch_size=config.batch_size, shuffle=False, num_workers=NUM_WORKERS,
                                 pin_memory=True, drop_last=False, persistent_workers=NUM_WORKERS > 0)
        results = []
        with t.no_grad():
            for img, img_names in tqdm(test_loader):
                img = img.to(device)
                pred_cls, _ = res_net(img)
                results += [[name, code] for name, code in zip(img_names, parse2class(pred_cls))]

    results = sorted(results, key=lambda x: x[0])
    write2csv(results, csv_path)
    return results


def ensemble_predict(model_paths, csv_path, model_type=None):
    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
    mt = model_type or config.model_type
    models = []
    for mp in model_paths:
        m = create_model(mt).to(device)
        ckpt = t.load(mp, map_location=device, weights_only=False)
        m.load_state_dict(ckpt['model'])
        m.eval()
        models.append(m)
        print(f'Loaded model: {mp}')

    test_set_for_count = DigitsDataset(mode='test', aug=False,
                                       input_size=(config.input_height, config.input_width))
    num_test = len(test_set_for_count)
    all_probs = [t.zeros(num_test, config.class_num) for _ in range(config.num_heads)]
    all_names = []

    for tta_size in config.tta_sizes:
        test_set_tta = DigitsDataset(mode='test', aug=False,
                                     input_size=(tta_size, tta_size))
        test_loader_tta = DataLoader(test_set_tta, batch_size=config.batch_size,
                                     shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
                                     drop_last=False, persistent_workers=False)
        sample_idx = 0
        with t.no_grad():
            for img, img_names in tqdm(test_loader_tta, desc=f'Ensemble TTA size={tta_size}'):
                img = img.to(device)
                bs = img.size(0)
                for model in models:
                    probs = model.forward_with_probs(img)
                    for h in range(config.num_heads):
                        all_probs[h][sample_idx:sample_idx + bs] += probs[h].cpu()
                if tta_size == config.tta_sizes[0]:
                    all_names.extend(img_names)
                sample_idx += bs

    results = [[name, code] for name, code in zip(all_names, parse2class_from_probs(all_probs))]
    results = sorted(results, key=lambda x: x[0])
    write2csv(results, csv_path)
    return results


def ctc_predict(model_path, csv_path, use_tta=False):
    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
    model = CTCModel(num_classes=config.class_num).to(device)
    model.load_state_dict(t.load(model_path, map_location=device, weights_only=False)['model'])
    print('Load CTC model from %s successfully' % model_path)
    model.eval()
    char_list = [str(i) for i in range(10)] + ['']

    test_loader = DataLoader(CTCDataset(mode='test', aug=False,
                                        input_size=(config.input_height, config.input_width)),
                             batch_size=config.batch_size, shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=True, drop_last=False, persistent_workers=NUM_WORKERS > 0,
                             collate_fn=ctc_test_collate_fn)
    results = []
    with t.no_grad():
        for img, img_names in tqdm(test_loader, desc='CTC Predict'):
            img = img.to(device)
            log_probs = model(img)
            pred_strs = ctc_beam_decode(log_probs.cpu())
            for b in range(len(img_names)):
                code = ''.join(char_list[d] for d in pred_strs[b])
                results.append([img_names[b], code])

    results = sorted(results, key=lambda x: x[0])
    write2csv(results, csv_path)
    return results


def cross_model_ensemble(multihead_path, ctc_path, csv_path, model_type=None):
    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
    mt = model_type or config.model_type
    mh_model = create_model(mt).to(device)
    mh_model.load_state_dict(t.load(multihead_path, map_location=device, weights_only=False)['model'])
    mh_model.eval()
    print(f'Loaded multihead model: {multihead_path}')

    ctc_model = CTCModel(num_classes=config.class_num).to(device)
    ctc_model.load_state_dict(t.load(ctc_path, map_location=device, weights_only=False)['model'])
    ctc_model.eval()
    print(f'Loaded CTC model: {ctc_path}')

    mh_test_set = DigitsDataset(mode='test', aug=False,
                                input_size=(config.input_height, config.input_width))
    ctc_test_set = CTCDataset(mode='test', aug=False,
                              input_size=(config.input_height, config.input_width))
    assert len(mh_test_set) == len(ctc_test_set), \
        f'Multihead test set ({len(mh_test_set)}) != CTC test set ({len(ctc_test_set)})'

    mh_test_loader = DataLoader(mh_test_set, batch_size=config.batch_size,
                                shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
                                drop_last=False, persistent_workers=False)
    ctc_test_loader = DataLoader(ctc_test_set, batch_size=config.batch_size,
                                 shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
                                 drop_last=False, persistent_workers=False,
                                 collate_fn=ctc_test_collate_fn)

    char_list = [str(i) for i in range(10)] + ['']
    results = []

    with t.no_grad():
        for (mh_img, img_names), (ctc_img, _) in tqdm(
                zip(mh_test_loader, ctc_test_loader), desc='Cross-model Ensemble',
                total=len(mh_test_loader)):
            mh_img = mh_img.to(device)
            ctc_img = ctc_img.to(device)
            bs = mh_img.size(0)

            mh_probs = mh_model.forward_with_probs(mh_img)
            mh_preds = [p.argmax(1) for p in mh_probs]
            mh_confidences = [p.max(1)[0] for p in mh_probs]

            ctc_log_probs = ctc_model(ctc_img)
            ctc_preds = ctc_beam_decode(ctc_log_probs.cpu())
            ctc_probs = ctc_log_probs.exp().cpu()
            ctc_confidences = []
            for b in range(bs):
                conf = []
                prev = -1
                for step in range(ctc_probs.size(0)):
                    p = ctc_probs[step, b].argmax().item()
                    if p != prev and p != 10:
                        conf.append(ctc_probs[step, b, p].item())
                    prev = p
                ctc_confidences.append(conf)

            for b in range(bs):
                mh_code = ''.join(char_list[mh_preds[h][b].item()] for h in range(config.num_heads))
                ctc_code = ''.join(char_list[d] for d in ctc_preds[b])

                if mh_code == ctc_code:
                    final_code = mh_code
                else:
                    mh_digits = []
                    for h in range(config.num_heads):
                        d = mh_preds[h][b].item()
                        if d == 10:
                            break
                        mh_digits.append((str(d), mh_confidences[h][b].item()))

                    ctc_digits = [(char_list[d], ctc_confidences[b][i] if i < len(ctc_confidences[b]) else 0.0)
                                  for i, d in enumerate(ctc_preds[b])]

                    max_len = max(len(mh_digits), len(ctc_digits))
                    final_chars = []
                    for pos in range(max_len):
                        mh_c, mh_conf = mh_digits[pos] if pos < len(mh_digits) else ('', 0.0)
                        ctc_c, ctc_conf = ctc_digits[pos] if pos < len(ctc_digits) else ('', 0.0)
                        if mh_c == ctc_c:
                            final_chars.append(mh_c)
                        elif mh_conf >= ctc_conf:
                            final_chars.append(mh_c)
                        else:
                            final_chars.append(ctc_c)
                    final_code = ''.join(final_chars)

                results.append([img_names[b], final_code])

    results = sorted(results, key=lambda x: x[0])
    write2csv(results, csv_path)
    return results
