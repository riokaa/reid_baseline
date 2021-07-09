# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os.path as osp

# import re

from PIL import Image
from torch.utils.data import Dataset


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert("RGB")
            got_img = True
        except IOError:
            print(
                "IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(
                    img_path
                )
            )
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None, with_index=False):
        self.dataset = dataset
        self.transform = transform
        self.with_index = with_index

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        if self.with_index:
            return img, pid, camid, img_path, index
        return img, pid, camid, img_path

    # def get_pids(self, file_path):
    #     """Suitable for muilti-dataset training"""
    #     if "cuhk03" in file_path:
    #         prefix = "cuhk"
    #         pid = "_".join(file_path.split("/")[-1].split("_")[0:2])
    #     else:
    #         prefix = file_path.split("/")[1]
    #         pat = re.compile(r"([-\d]+)_c(\d)")
    #         pid, _ = pat.search(file_path).groups()
    #     return prefix + "_" + pid
