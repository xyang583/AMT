import enum
import torch
import numpy as np
import cv2
from PIL import Image
import torch.distributed as dist


def worker_init_fn_(worker_id, seed):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    rank = dist.get_rank()
    random_gen = np.random.RandomState(seed + worker_id + rank)
    dataset.random_gen = random_gen
    for d in dataset.dataset_list:
        d.random_gen = random_gen


def cycle_(iterable):


    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


class Split(enum.Enum):
    """The possible data splits."""
    TRAIN = 0
    VALID = 1
    TEST = 2


def parse_record(feat_dic):






    image = cv2.imdecode(feat_dic["image"], -1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    feat_dic["image"] = image
    return feat_dic


