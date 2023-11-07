from __future__ import absolute_import, division, print_function
import torch
import numpy as np
import random

import re

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging"""
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)
    
def f_in_hook(l):
    return lambda m, i, o: l.append(i)
    
def f_out_hook(l):
    return lambda m, i, o: l.append(o)
    
def f_in_out_hook(li, lo):
    def fn(m, i, o):
        li.append(i)
        lo.append(o)
    return fn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
def load_vocab_dict_from_file(dict_file):
    with open(dict_file) as f:
        words = [w.strip() for w in f.readlines()]
    vocab_dict = {words[n]:n for n in range(len(words))}
    return vocab_dict

UNK_IDENTIFIER = '<unk>' # <unk> is the word used to identify unknown words
SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
def sentence2vocab_indices(sentence, vocab_dict):
    words = SENTENCE_SPLIT_REGEX.split(sentence.strip())
    words = [w.lower() for w in words if len(w.strip()) > 0]
    # remove .
    if words[-1] == '.':
        words = words[:-1]
    vocab_indices = [(vocab_dict[w] if w in vocab_dict else vocab_dict[UNK_IDENTIFIER])
        for w in words]
    return vocab_indices

PAD_IDENTIFIER = '<pad>'
EOS_IDENTIFIER = '<eos>'
def preprocess_sentence(sentence, vocab_dict, T):
    vocab_indices = sentence2vocab_indices(sentence, vocab_dict)
    # # Append '<eos>' symbol to the end
    # vocab_indices.append(vocab_dict[EOS_IDENTIFIER])
    # Truncate long sentences
    if len(vocab_indices) > T:
        vocab_indices = vocab_indices[:T]
    # Pad short sentences at the beginning with the special symbol '<pad>'
    if len(vocab_indices) < T:
        vocab_indices = [vocab_dict[PAD_IDENTIFIER]] * (T - len(vocab_indices)) + vocab_indices
    return vocab_indices
