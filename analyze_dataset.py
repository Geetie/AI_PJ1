import json
import os
import numpy as np

def analyze_dataset(label_path, name):
    with open(label_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    lengths = []
    chars = [0]*11  # 0-9, 10=empty
    
    for img_name, mark in labels.items():
        num_chars = len(mark['label'])
        lengths.append(num_chars)
        
        for c in mark['label']:
            if 0 <= c <= 10:
                chars[c] += 1
            else:
                print(f"Warning: invalid char {c} in {img_name}")
    
    lengths = np.array(lengths)
    chars = np.array(chars)
    
    print(f"=== {name} Dataset ===")
    print(f"Total: {len(labels)} images")
    print(f"Length distribution:")
    for l in range(0, 7):
        count = (lengths == l).sum()
        ratio = count / len(labels) * 100
        print(f"  Length {l}: {count} ({ratio:.2f}%)")
    print(f"Length mean: {lengths.mean():.2f}, std: {lengths.std():.2f}")
    print(f"Char distribution:")
    for c in range(0, 11):
        if c == 10:
            name = 'empty'
        else:
            name = str(c)
        total = chars.sum()
        ratio = chars[c] / total * 100 if total > 0 else 0
        print(f"  {name}: {chars[c]} ({ratio:.2f}%)")
    print()
    return lengths, chars

if __name__ == '__main__':
    from config import data_dir
    train_lengths, train_chars = analyze_dataset(data_dir['train_label'], "Train")
    val_lengths, val_chars = analyze_dataset(data_dir['val_label'], "Val")
