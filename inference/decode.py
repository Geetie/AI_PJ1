import numpy as np
import torch as t


def parse2class(prediction):
    char_list = [str(i) for i in range(10)] + ['']
    preds = [p.argmax(1) for p in prediction]
    batch_size = preds[0].shape[0]
    res = []
    for b in range(batch_size):
        chars = []
        for h in range(len(preds)):
            idx = preds[h][b].item()
            if idx == 10:
                break
            chars.append(char_list[idx])
        res.append(''.join(chars))
    return res


def parse2class_from_probs(all_probs):
    char_list = [str(i) for i in range(10)] + ['']
    res = []
    for b in range(len(all_probs[0])):
        chars = []
        for h in range(len(all_probs)):
            idx = all_probs[h][b].argmax().item()
            if idx == 10:
                break
            chars.append(char_list[idx])
        res.append(''.join(chars))
    return res


def ctc_greedy_decode(log_probs):
    log_probs = log_probs.permute(1, 0, 2)
    results = []
    for b in range(log_probs.size(0)):
        pred = log_probs[b].argmax(dim=1).tolist()
        decoded = []
        prev = -1
        for p in pred:
            if p != prev and p != 10:
                decoded.append(p)
            prev = p
        results.append(decoded)
    return results


def ctc_beam_decode(log_probs, beam_width=10):
    log_probs = log_probs.permute(1, 0, 2)
    B, T, C = log_probs.shape
    blank = 10
    results = []
    for b in range(B):
        beam_b = {}
        beam_nb = {}
        beam_b[((),)] = 0.0
        beam_nb[((),)] = -float('inf')
        for t in range(T):
            new_beam_b = {}
            new_beam_nb = {}
            topk_vals, topk_idx = log_probs[b, t].topk(min(beam_width, C))
            for k in range(topk_idx.size(0)):
                c = topk_idx[k].item()
                log_p = topk_vals[k].item()
                for prefix in set(list(beam_b.keys()) + list(beam_nb.keys())):
                    if c == blank:
                        score = max(beam_b.get(prefix, -float('inf')),
                                    beam_nb.get(prefix, -float('inf')))
                        if score > -float('inf'):
                            if prefix in new_beam_b:
                                new_beam_b[prefix] = np.logaddexp(new_beam_b[prefix], score + log_p)
                            else:
                                new_beam_b[prefix] = score + log_p
                    else:
                        new_prefix = prefix + (c,)
                        last_c = prefix[-1] if len(prefix) > 0 else None
                        if c == last_c:
                            score_b = beam_b.get(prefix, -float('inf'))
                            if score_b > -float('inf'):
                                if new_prefix in new_beam_nb:
                                    new_beam_nb[new_prefix] = np.logaddexp(new_beam_nb[new_prefix], score_b + log_p)
                                else:
                                    new_beam_nb[new_prefix] = score_b + log_p
                            score_nb = beam_nb.get(prefix, -float('inf'))
                            if score_nb > -float('inf'):
                                same_prefix_score = score_nb + log_p
                                if prefix in new_beam_nb:
                                    new_beam_nb[prefix] = np.logaddexp(new_beam_nb[prefix], same_prefix_score)
                                else:
                                    new_beam_nb[prefix] = same_prefix_score
                        else:
                            score = max(beam_b.get(prefix, -float('inf')),
                                        beam_nb.get(prefix, -float('inf')))
                            if score > -float('inf'):
                                if new_prefix in new_beam_nb:
                                    new_beam_nb[new_prefix] = np.logaddexp(new_beam_nb[new_prefix], score + log_p)
                                else:
                                    new_beam_nb[new_prefix] = score + log_p
            all_beams = {}
            for p, s in new_beam_b.items():
                all_beams[p] = all_beams.get(p, -float('inf'))
                all_beams[p] = np.logaddexp(all_beams[p], s)
            for p, s in new_beam_nb.items():
                all_beams[p] = all_beams.get(p, -float('inf'))
                all_beams[p] = np.logaddexp(all_beams[p], s)
            sorted_beams = sorted(all_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width]
            beam_b = {}
            beam_nb = {}
            for prefix, score in sorted_beams:
                beam_b[prefix] = new_beam_b.get(prefix, -float('inf'))
                beam_nb[prefix] = new_beam_nb.get(prefix, -float('inf'))
        all_final = {}
        for p, s in beam_b.items():
            all_final[p] = all_final.get(p, -float('inf'))
            all_final[p] = np.logaddexp(all_final[p], s)
        for p, s in beam_nb.items():
            all_final[p] = all_final.get(p, -float('inf'))
            all_final[p] = np.logaddexp(all_final[p], s)
        if all_final:
            best = max(all_final, key=all_final.get)
            results.append(list(best))
        else:
            results.append([])
    return results
