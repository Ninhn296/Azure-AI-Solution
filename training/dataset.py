#!/usr/bin/env python
import os
import numpy as np
import torch
from PIL import Image
import xml.etree.ElementTree as ET
import cv2
from model_ssd.ssd import MatchPrior
from model_ssd import mobilenetv1_ssd_config


config = mobilenetv1_ssd_config

target_transform = MatchPrior(config.priors,
                              config.center_variance,
                              config.size_variance,
                              0.5)


class VocDataset(object):
    """ DataSet with VOC Pascal format support
    """

    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        temp_annos = list(sorted(os.listdir(os.path.join(root,
                                                         "Annotations"))))
        self.imgs = []
        self.annos = []

        for file in temp_annos:
            path = os.path.join(self.root, "Annotations", file)
            anno_infor = ET.parse(path)
            objs = anno_infor.findall("object")
            if objs:
                self.annos.append(file)
                self.imgs.append(file.split(".")[0] + '.jpg')

        self.num_classes = []
        self.classes_path = os.path.join(root, "class_names.txt")
        with open(self.classes_path, "r") as fr:
            self.num_classes = fr.read().splitlines()

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root,
                                "JPEGImages",
                                self.imgs[idx])
        anno_path = os.path.join(self.root,
                                 "Annotations",
                                 self.annos[idx])

        # Get label list
        labels = []
        boxes = []
        anno_infor = ET.parse(anno_path)
        objs = anno_infor.findall("object")

        for obj in objs:
            obj_name = obj.find("name").text
            class_id = self.num_classes.index(obj_name)
            labels.append(class_id)
            xmin = obj.find("bndbox/xmin").text
            xmax = obj.find("bndbox/xmax").text
            ymin = obj.find("bndbox/ymin").text
            ymax = obj.find("bndbox/ymax").text
            boxes.append([float(xmin)-1, float(ymin)-1, float(xmax)-1, float(ymax)-1])

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
            img, boxes, labels = self.transforms(img,
                                                 boxes,
                                                 labels)
            boxes, labels = target_transform(boxes, labels)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        # For evaluate
        num_objs = len(objs)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        return img, target

    def __len__(self):
        return len(self.imgs)
