import torch
import torch.utils.data as data
from pycocotools.coco import COCO
import pandas as pd
import numpy as np
import torch

from .transforms import get_transform


class MNISTDataset(data.Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = pd.read_csv(self.dataset_path).values

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index, 1:].reshape((1, 28, 28))).float(),\
               torch.LongTensor([self.data[index, 0]])


class COCODetectionDataset(data.Dataset):
    """
    MS Coco Detection Dataset.
    http://cocodataset.org/#format-data
    Args:
        root (string): Root directory where images are downloaded to.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
    """

    def __init__(self, ann_path, imgs_path, transform=None):
        self.imgs_path = imgs_path
        self.coco = COCO(ann_path)
        self.transform = transform

        self.ids = list(self.coco.imgToAnns.keys())

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``coco.loadAnns``.
        """
        img = self.pull_image(index)
        bboxes, labels = self.pull_annotation(index)

        if self.transform is not None:
            transformed = self.transform(image=img, bboxes=bboxes, category_id=labels)
            img = transformed["image"]
            bboxes = transformed["bboxes"]
            labels = transformed["category_id"]

        img = self._preprocess(img)
        _, height, width = img.shape
        bboxes = np.array(bboxes)
        bboxes[:, 2] += bboxes[:, 0]
        bboxes[:, 3] += bboxes[:, 1]
        bboxes = bboxes / np.array([width, height, width, height], dtype=np.float)  # normalize
        target = np.hstack([bboxes, np.expand_dims(labels, axis=1)])

        return torch.from_numpy(img), target

    def pull_image(self, index):
        """
        Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        """
        img_id = self.ids[index]
        img = cv2.imread(osp.join(self.imgs_path, self.coco.loadImgs(img_id)[0]["file_name"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def pull_annotation(self, index):
        """
        Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            bboxes:
            categories:
        """
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        bboxes, categories = [], []
        for ann in anns:
            bboxes.append(ann['bbox'])
            categories.append(ann["category_id"] - 1)
        return bboxes, categories

    @staticmethod
    def _preprocess(img):
        img = np.transpose(img, (2, 0, 1))
        return img


def get_dataset(config):
    if config['dataset'] == 'MNIST':
        dataset = MNISTDataset(config['dataset_path'])
    elif config['dataset'] == 'COCO':
        transform = get_transform(config['img_size'], transform_type=config['transform'])
        dataset = COCODetectionDataset(config["ann_path"], config["img_path"], transform=transform)
    else:
        raise ValueError("Dataset [%s] not recognized." % config['dataset'])
    return dataset
