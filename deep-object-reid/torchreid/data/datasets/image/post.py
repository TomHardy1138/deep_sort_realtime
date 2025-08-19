from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp

from ..dataset import ImageDataset
from torchvision.datasets import ImageFolder


class SimilarityDatasetCommon(ImageDataset):
    dataset_dir = "common" # "SimilarityDatasetCommon"
    dataset_url = None

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        images = ImageFolder(self.dataset_dir)

        train = []
        label = {}
        label_count = -1

        for image, _ in images.imgs:
            label_tmp = osp.dirname(image)
            if label_tmp not in label.keys():
                label_count += 1
                label[label_tmp] = label_count
                lb = label_count
            else:
                lb = label[label_tmp]
            train.append((image, lb, 0))
        
        super(SimilarityDatasetCommon, self).__init__(train, [], [], **kwargs)


class SimilarityDatasetCommonTest(ImageDataset):
    dataset_dir = "common_test"
    dataset_url = None

    def __init__(self, root='', **kwargs):

        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        images = ImageFolder(self.dataset_dir)

        gpids = []

        query, gallery = [], []

        for image, label in images.imgs:
            if label not in gpids:
                gpids.append(label)
                gallery.append((image, label, 0))
            else:
                query.append((image, label, len(query) + 2))

        # query, gallery = gallery, query

        super(SimilarityDatasetCommonTest, self).__init__([], query, gallery, **kwargs)


class Food(ImageDataset):
    dataset_dir = 'food'
    dataset_url = None

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        images = ImageFolder(self.dataset_dir)

        query, gallery, train = [], [], []

        for image, label in images.imgs:
            train.append((image, label, 0))

        super(Food, self).__init__(train, query, gallery, **kwargs)


class Food_Test(ImageDataset):
    dataset_dir = 'food_test'
    dataset_url = None

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        images = ImageFolder(self.dataset_dir)

        gpids = []

        query, gallery = [], []

        for image, label in images.imgs:
            if label not in gpids:
                gpids.append(label)
                gallery.append((image, label, 0))
            else:
                query.append((image, label, len(query) + 2))

        train = []

        query, gallery = gallery, query

        super(Food_Test, self).__init__(train, query, gallery, **kwargs)


class NEWPOST(ImageDataset):
    dataset_dir = 'new_post'
    dataset_url = None

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        images = ImageFolder(self.dataset_dir)

        query, gallery, train = [], [], []

        for image, label in images.imgs:
            train.append((image, label, 0))

        super(NEWPOST, self).__init__(train, query, gallery, **kwargs)


class NEWPOST_Test(ImageDataset):
    dataset_dir = "DPE-2878_new"#'new_post_test'
    dataset_url = None

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        images = ImageFolder(self.dataset_dir)

        gpids = []

        query, gallery = [], []

        import random as rd
        imgs = images.imgs
        rd.shuffle(imgs)

        for image, label in imgs[:20000]:
            if label not in gpids:
                gpids.append(label)
                gallery.append((image, label, 0))
            else:
                query.append((image, label, len(query) + 2))

        train = []

        # query, gallery = gallery, query

        super(NEWPOST_Test, self).__init__(train, query, gallery, **kwargs)
