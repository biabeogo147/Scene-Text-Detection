import os
import torch
from torch.nn import functional as F


def encode(label, char2idx, max_label_len):
    encoded_labels = torch.tensor([char2idx[char] for char in label], dtype=torch.long)
    label_len = torch.tensor(len(encoded_labels), dtype=torch.long)
    padded_labels = F.pad(encoded_labels, (0, max_label_len - len(encoded_labels)), value=0)
    return padded_labels, label_len


def decode(encoded_sequences, idx2char, blank_char='-'):
    decoded_sequences = []
    for seq in encoded_sequences:
        decoded_label = []
        prev_char = None

        for token in seq:
            if token != 0: # Ignore padding (token=0)
                char = idx2char[token.item()]
                # Append char if it's not a blank and not the same as the prev char
                if char != blank_char:
                    if char != prev_char or prev_char == blank_char:
                        decoded_label.append(char)
                prev_char = char

        decoded_sequences.append(''.join(decoded_label))
    return decoded_sequences


def build_vocab(im_paths, im_labels):
    save_dir = './datasets/ocr_dataset'
    with open(os.path.join(save_dir, 'labels.txt'), 'r') as f:
        for label in f:
            im_labels.append(label.strip().split('\t')[1])
            im_paths.append(label.strip().split('\t')[0])

    letters = [char.split('.')[0].lower() for char in im_labels]
    letters = ''.join(letters)
    letters = sorted(set(letters))

    chars = ''.join(letters)
    blank_char = '-'
    chars += blank_char
    vocab_size = len(chars)

    print(f'Vocab size: {vocab_size}')
    print(chars)

    char2idx = {char: idx + 1 for idx, char in enumerate(sorted(chars))}
    idx2char = {idx: char for char, idx in char2idx.items()}

    max_label_len = max([len(label) for label in im_labels])
    print(max_label_len)

    return char2idx, idx2char, max_label_len, vocab_size, blank_char