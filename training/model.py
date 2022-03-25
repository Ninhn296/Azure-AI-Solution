#!/usr/bin/env python
import torchvision
from model_ssd.mobilenetv1_ssd import create_mobilenetv1_ssd


class Model(object):

    def __init__(self):
        pass

    def create_SSD(self, num_classes, is_test=False, is_init=False):
        """
            Create the model that will be used to training
            Args:
                num_classes(int): number of class, that model can detect
            Returns:
                model: model that will be used to training
        """
        model = create_mobilenetv1_ssd(num_classes, is_test=is_test)
        # Load pretrained model
        if is_init:
            model.init_from_pretrained_ssd('./pretrained/mobilenet-v1-ssd-mp-0_675.pth')

        return model
