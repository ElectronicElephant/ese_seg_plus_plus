import os
import numpy as np
from PIL import Image
import torch
import argparse

coco_2017_cat = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck',
                 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
                 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
                 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
                 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
                 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
                 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza',
                 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
                 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
                 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
                 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}


def readBases(path, cat_dict, cat_id, scale=(64, 64)):
    """
    From Wenqiang
    """
    path += ("_" + str(scale[0]) + "_" + str(scale[1]))
    base_list = os.listdir(path + "/" + cat_dict[cat_id])
    all_bases = np.zeros((len(base_list), scale[0], scale[1]))
    for i in range(len(base_list)):
        basis = np.array(Image.open(path + "/" + cat_dict[cat_id] + "/" + base_list[i]))
        all_bases[i] = basis.copy()
    return all_bases


class ScatterWrapper:
    """ Input is any number of lists. This will preserve them through a dataparallel scatter. """

    # Derived from coco 2017 dataset
    bases_dict = coco_2017_cat

    def __init__(self, *args):
        pass

    def set_bases(self, base):
        ScatterWrapper.bases_dict = base

    @staticmethod
    def get_bases(cat_id):
        all_bases = np.zeros((50, 64, 64))
        for i in range(50):
            all_bases[i] = ScatterWrapper.bases_dict[cat_id][i]
        all_bases = np.transpose(all_bases, (1, 2, 0))
        # print(all_bases.shape)
        return all_bases


def main():
    # Read bases
    bases_loc = r'data/bases'
    all_bases = {}
    print('Loading predefined bases...')
    for cat_id_ in coco_2017_cat.keys():
        bases = readBases(bases_loc, coco_2017_cat, cat_id=cat_id_, scale=(64, 64))
        all_bases[cat_id_] = (bases.copy())

    wrapper = ScatterWrapper([])
    wrapper.set_bases(all_bases)

    a = wrapper.get_bases(1).reshape(50, 64, 64)
    # print(a.shape)

    # [xmin, ymin, xmax, ymax, label_idx]
    gt_box = [0.5, 0.2, 1, 1]
    scale = 138

    xmin, ymin, xmax, ymax = [round(x * scale) for x in gt_box]

    all_bases = np.zeros((50, 138, 138))
    for i in range(50):
        img = Image.fromarray(a[0])
        img = img.resize(((xmax - xmin), (ymax - ymin)))
        img = np.array(img)
        # print(img.shape)
        img = np.pad(img, ((ymin, scale - ymax), (xmin, scale - xmax)), 'constant', constant_values=(0, 0))
        # print(img.shape)
        all_bases[i] = img.copy()

    # print(all_bases.shape)


if __name__ == '__main__':
    a = np.ones((3, 3, 5))
    a[0, 0, :] = np.array((2, 2, 2, 2, 2))
    print(a)
    b = np.array([1, 2, 3, 4, 5])
    print(a @ b)

    main()
