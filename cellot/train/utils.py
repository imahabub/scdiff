import torch
from pathlib import Path
from absl import logging
from cellot.utils.helpers import flat_dict, nest_dict
import GPUtil
import os

def check_loss(*args):
    for arg in args:
        if torch.isnan(arg):
            raise ValueError


def load_item_from_save(path, key, default):
    path = Path(path)
    if not path.exists():
        return default

    ckpt = torch.load(path)
    if key not in ckpt:
        logging.warn(f'\'{key}\' not found in ckpt: {str(path)}')
        return default

    return ckpt[key]


def cast_loader_to_iterator(loader, cycle_all=True):
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    iterator = nest_dict({
        key: cycle(item)
        for key, item
        in flat_dict(loader).items()
    }, as_dot_dict=True)

    return iterator

def get_free_gpu():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    # Set environment variables for which GPUs to use.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    chosen_gpu = ''.join(
        [str(x) for x in GPUtil.getAvailable(order='memory')])
    # os.environ["CUDA_VISIBLE_DEVICES"] = chosen_gpu
    print(f"Using GPUs: {chosen_gpu}")
    return chosen_gpu