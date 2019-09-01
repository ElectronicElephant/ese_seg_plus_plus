import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
from .config import cfg
from pycocotools import mask as maskUtils

def get_label_map():
    if cfg.dataset.label_map is None:
        return {x+1: x+1 for x in range(len(cfg.dataset.class_names))}
    else:
        return cfg.dataset.label_map 

class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """
    def __init__(self):
        self.label_map = get_label_map()

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                label_idx = self.label_map[obj['category_id']] - 1
                final_box = list(np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])/scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                print("No bbox found for object ", obj)

        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]


class COCODetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
        prep_crowds (bool): Whether or not to prepare crowds for the evaluation step.
    """

    def __init__(self, image_path, info_file, transform=None,
                 target_transform=COCOAnnotationTransform(),
                 dataset_name='MS COCO', has_gt=True):
        # Do this here because we have too many things named COCO
        from pycocotools.coco import COCO
        
        self.root = image_path
        self.coco = COCO(info_file)
        
        self.ids = list(self.coco.imgToAnns.keys())
        if len(self.ids) == 0 or not has_gt:
            self.ids = list(self.coco.imgs.keys())
        
        self.transform = transform
        self.target_transform = target_transform
        
        self.name = dataset_name
        self.has_gt = has_gt

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, (target, masks, num_crowds)).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt, masks, h, w, num_crowds = self.pull_item(index)
        return im, (gt, masks, num_crowds)

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, masks, height, width, crowd).
                   target is the object returned by ``coco.loadAnns``.
            Note that if no crowd annotations exist, crowd will be None
        """
        img_id = self.ids[index]

        if self.has_gt:
            target = self.coco.imgToAnns[img_id]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)

            # Target has {'segmentation', 'area', iscrowd', 'image_id', 'bbox', 'category_id'}
            target = self.coco.loadAnns(ann_ids)
            # [{'segmentation': [[176.28, 17.9, 187.29, 17.9, 199.23, 14.23, 241.46, 24.33, 273.59, 30.76, 310.32, 0.46, 440.69, 0.46, 463.64, 33.51, 390.19, 52.79, 313.07, 70.23, 233.2, 85.84, 219.43, 84.01, 188.21, 42.69]], 'area': 14543.05615, 'iscrowd': 0, 'image_id': 468951, 'bbox': [176.28, 0.46, 287.36, 85.38], 'category_id': 17, 'id': 46331}, {'segmentation': [[572.47, 0.96, 247.94, 83.29, 180.93, 89.99, 136.89, 108.18, 114.88, 134.02, 107.22, 169.44, 117.75, 196.25, 126.36, 215.39, 134.02, 266.13, 153.17, 310.17, 153.17, 350.37, 154.13, 374.31, 163.7, 381.01, 165.61, 408.77, 170.4, 419.3, 636.61, 421.21, 639.48, 0.96, 570.55, 1.91]], 'area': 186528.08160000003, 'iscrowd': 0, 'image_id': 468951, 'bbox': [107.22, 0.96, 532.26, 420.25], 'category_id': 33, 'id': 1183135}][{'segmentation': [[0.92, 29.84, 21.12, 27.08, 47.74, 43.61, 48.66, 76.66, 54.17, 95.94, 73.45, 102.37, 95.48, 103.29, 104.66, 117.98, 96.4, 138.17, 89.06, 151.03, 93.65, 162.05, 461.81, 159.29, 470.99, 140.01, 464.56, 128.08, 471.91, 121.65, 475.58, 95.02, 462.72, 78.5, 435.18, 73.91, 404.88, 74.83, 399.38, 61.05, 417.74, 44.53, 491.19, 34.43, 546.27, 27.08, 567.39, 22.49, 592.18, 40.86, 592.18, 57.38, 569.22, 90.43, 556.37, 99.61, 525.16, 101.45, 515.06, 96.86, 509.55, 100.53, 499.45, 103.29, 493.02, 127.16, 509.55, 136.34, 513.22, 153.78, 539.84, 155.62, 571.98, 154.7, 581.16, 152.86, 597.69, 161.13, 638.08, 164.8, 639.0, 414.52, 639.0, 424.16, 482.92, 426.0, 256.15, 421.41, 40.4, 422.33, 2.75, 425.08, 0.92, 235.95, 1.84, 32.13]], 'area': 188763.05045, 'iscrowd': 0, 'image_id': 302029, 'bbox': [0.92, 22.49, 638.08, 403.51], 'category_id': 2, 'id': 125131}, {'segmentation': [[191.36, 187.28, 181.75, 169.97, 181.75, 161.31, 212.52, 161.31, 248.1, 165.16, 288.49, 163.24, 327.91, 162.27, 374.07, 162.27, 428.88, 160.35, 467.35, 152.66, 538.51, 148.81, 558.7, 144.0, 633.71, 160.35, 640.0, 168.04, 629.86, 182.47, 599.09, 182.47, 491.39, 206.51, 445.23, 210.36, 384.65, 210.36, 323.11, 208.43, 269.26, 208.43, 228.87, 205.55, 197.13, 201.7, 182.71, 201.7, 163.48, 201.7, 125.01, 199.78, 96.16, 218.05, 110.59, 240.17, 164.44, 251.71, 158.67, 262.28, 127.9, 264.21, 101.93, 259.4, 81.74, 244.01, 89.43, 195.93, 103.86, 186.31]], 'area': 23708.748499999994, 'iscrowd': 0, 'image_id': 302029, 'bbox': [81.74, 144.0, 558.26, 120.21], 'category_id': 28, 'id': 279922}, {'segmentation': [[402.07, 206.78, 403.98, 218.27, 414.51, 227.84, 433.66, 236.45, 440.36, 254.64, 446.1, 269.0, 439.4, 300.59, 435.57, 334.1, 434.62, 347.5, 424.09, 351.33, 423.13, 339.84, 415.47, 329.31, 405.9, 330.27, 401.11, 343.67, 392.49, 351.33, 391.54, 365.69, 391.54, 381.01, 390.58, 400.15, 386.75, 416.43, 380.05, 420.26, 352.29, 420.26, 318.78, 419.3, 318.78, 414.51, 336.97, 400.15, 346.54, 392.49, 343.67, 369.52, 311.12, 345.59, 307.29, 334.1, 309.21, 325.48, 302.51, 320.7, 290.06, 327.4, 290.06, 342.71, 276.66, 343.67, 261.34, 348.46, 255.6, 352.29, 249.86, 345.59, 246.98, 338.89, 244.11, 326.44, 244.11, 308.25, 247.94, 297.72, 255.6, 293.89, 272.83, 288.15, 284.32, 277.62, 294.85, 267.09, 299.64, 249.86, 298.68, 235.5, 293.89, 224.97, 291.98, 209.65, 318.78, 209.65, 351.33, 207.73, 381.96, 206.78], [467.16, 295.81, 478.65, 315.91, 460.46, 328.36, 465.25, 295.81]], 'area': 25827.640750000002, 'iscrowd': 0, 'image_id': 302029, 'bbox': [244.11, 206.78, 234.54, 213.48], 'category_id': 1, 'id': 1720662}]
            # 这里的 BBOX 依然是正常的
        else:
            target = []

        # Separate out crowd annotations. These are annotations that signify a large crowd of
        # objects of said class, where there is no annotation for each individual object. Both
        # during testing and training, consider these crowds as neutral.
        crowd  = [x for x in target if     ('iscrowd' in x and x['iscrowd'])]
        target = [x for x in target if not ('iscrowd' in x and x['iscrowd'])]
        num_crowds = len(crowd)

        # This is so we ensure that all crowd annotations are at the end of the array
        target += crowd
        
        # The split here is to have compatibility with both COCO2014 and 2017 annotations.
        # In 2014, images have the pattern COCO_{train/val}2014_%012d.jpg, while in 2017 it's %012d.jpg.
        # Our script downloads the images as %012d.jpg so convert accordingly.
        file_name = self.coco.loadImgs(img_id)[0]['file_name']
        
        if file_name.startswith('COCO'):
            file_name = file_name.split('_')[-1]

        path = osp.join(self.root, file_name)
        assert osp.exists(path), 'Image path does not exist: {}'.format(path)
        
        img = cv2.imread(path)
        height, width, _ = img.shape
        
        if len(target) > 0:
            # Pool all the masks for this image into one [num_objects,height,width] matrix
            masks = [self.coco.annToMask(obj).reshape(-1) for obj in target]
            masks = np.vstack(masks)
            masks = masks.reshape(-1, height, width)

        if self.target_transform is not None and len(target) > 0:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            if len(target) > 0:
                target = np.array(target)
                img, masks, boxes, labels = self.transform(img, masks, target[:, :4],
                    {'num_crowds': num_crowds, 'labels': target[:, 4]})
            
                # I stored num_crowds in labels so I didn't have to modify the entirety of augmentations
                num_crowds = labels['num_crowds']
                labels     = labels['labels']
                
                target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            else:
                img, _, _, _ = self.transform(img, np.zeros((1, height, width), dtype=np.float), np.array([[0, 0, 1, 1]]),
                    {'num_crowds': 0, 'labels': np.array([0])})
                masks = None
                target = None

        return torch.from_numpy(img).permute(2, 0, 1), target, masks, height, width, num_crowds

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return cv2.imread(osp.join(self.root, path), cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        return self.coco.loadAnns(ann_ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
