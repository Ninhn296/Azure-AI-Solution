#!/usr/bin/env python
import os
import sys
import torch
from model import Model
from dataset import VocDataset
from model_ssd.transforms import *
from model_ssd import mobilenetv1_ssd_config
import itertools
from model_ssd.multibox_loss import MultiboxLoss
import shutil
sys.path.append('/opt/pytorch/vision/references/detection')
import utils
import transforms as T

CLASS_NAMES = "class_names.txt"

def get_transforms(train=True):
    """ Get transform object of the dataset.
        Args:
            train(bool): training status
        Returns:
            Transform object.
    """

    config = mobilenetv1_ssd_config
    if train:
        transforms = [ConvertFromInts(),
                      PhotometricDistort(),
                      Expand(config.image_mean),
                      RandomSampleCrop(),
                      RandomMirror(),
                      ToPercentCoords(),
                      Resize(config.image_size),
                      SubtractMeans(config.image_mean),
                      lambda img, boxes=None, labels=None: (img/config.image_std,
                                                            boxes,
                                                            labels),
                      ToTensor()]
    else:
        transforms = [ToPercentCoords(),
                      Resize(config.image_size),
                      SubtractMeans(config.image_mean),
                      lambda img, boxes=None, labels=None: (img/config.image_std,
                                                            boxes,
                                                            labels),
                      ToTensor()]
    return Compose(transforms)


def get_classes_num(path):
    """ Get class object of the dataset.
        Args:
            path(string):   path of dataset
        Returns:
            num_class(int) : number of class
    """
    with open(path, 'r') as f:
        lines = f.readlines()
    return len(lines)


def dataloaders(dataset_path):
    """ Load the dataset.
        Args:
            dataset_path(string):  dataset path
        Returns:
            dataset.
    """

    # use dataset and defined transformations
    train_dataset = VocDataset(dataset_path,
                               get_transforms(train=True))
    val_dataset = VocDataset(dataset_path,
                             get_transforms(train=False))

    train_data = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=1,
                                             shuffle=True,
                                             num_workers=0)
    val_data = torch.utils.data.DataLoader(val_dataset,
                                           batch_size=1,
                                           shuffle=False,
                                           num_workers=0)

    return train_data, val_data


def export_onnx_model(num_classes, model_name, onnx_model_name):
    """ Export model to onnx format.
        Args:
            num_classes: number of classes
            model_name(string): name of model.
            onnx_model_name(string): name of onnx model.
    """
    # Load model
    model = Model().create_SSD(num_classes, is_test=True)
    pretrained_dict = torch.load(model_name, map_location=torch.device('cuda'))
    model.load_state_dict(pretrained_dict)
    model.eval().cuda()

    # Export the model
    input_names = ["input"]
    output_names = ['scores', 'boxes']
    image_tensor = torch.randn(1, 3, 300, 300, requires_grad=True).cuda()
    torch.onnx.export(model,
                      image_tensor,
                      onnx_model_name,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=None)


def start_train(num_epochs, dataset_path, output):
    """ Perform training and evaluate
        Args:
            num_epochs(int):      number of epochs
            dataset_path(string): dataset path
            output(string):       output model path
        Returns:
            boolean:    training result.
    """

    class_names = os.path.join(dataset_path, CLASS_NAMES)
    num_classes = get_classes_num(class_names)

    # train on the GPU or on the CPU, if a GPU is not available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = Model().create_SSD(num_classes, is_init=True)

    # Dataloader
    train_data, val_data = dataloaders(dataset_path)

    # move model to the right device
    model.to(device)

    # Construct an optimizer
    lr = 0.002
    params = [
        {'params': model.base_net.parameters(), 'lr': lr},
        {'params': itertools.chain(
            model.source_layer_add_ons.parameters(),
            model.extras.parameters()
        ), 'lr': lr},
        {'params': itertools.chain(
            model.regression_headers.parameters(),
            model.classification_headers.parameters()
        )}
    ]
    optimizer = torch.optim.SGD(params, lr=lr,
                                momentum=0.9, weight_decay=0.0005)
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=20,
                                                   gamma=0.1)

    config = mobilenetv1_ssd_config
    criterion = MultiboxLoss(config.priors, iou_threshold=0.5,
                             neg_pos_ratio=3, center_variance=0.1,
                             size_variance=0.2, device=device)
    debug_steps = int(len(train_data)/10)
    is_success = True
    loss_info = []
    model.train()
    for epoch in range(num_epochs):
        try:
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0
            for i, data in enumerate(train_data):
                images, target = data
                images = images.to(device)
                boxes = target["boxes"].to(device)
                labels = target["labels"].to(device)
                confidence, locations = model(images)
                regression_loss, classification_loss = criterion(confidence,
                                                                 locations,
                                                                 labels,
                                                                 boxes)
                losses = regression_loss + classification_loss
                running_loss += losses.item()
                running_regression_loss += regression_loss.item()
                running_classification_loss += classification_loss.item()

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                if i and i % debug_steps == 0:
                    step = epoch + (i/debug_steps)/10
                    loss = running_loss/debug_steps
                    loss_classifier = running_classification_loss/debug_steps
                    loss_box_reg = running_regression_loss/debug_steps
                    loss_mask = 0
                    print(f"Epoch: {step} " +
                          f"Loss: {loss:.4f}, " +
                          f"loss_classifier {loss_classifier:.4f}, " +
                          f"loss_box_reg: {loss_box_reg:.4f}, " +
                          f"loss_mask: {loss_mask:.4f}")
                    loss_info.append([step, loss, loss_classifier, loss_box_reg, loss_mask])

                    running_loss = 0.0
                    running_regression_loss = 0.0
                    running_classification_loss = 0.0
            # update the learning rate
            lr_scheduler.step()
        except:
            is_success = False
            break

    if is_success:
        if not os.path.exists(output):
            os.mkdir(output)
        base = os.path.basename(output)
        out_classname = os.path.join(output, CLASS_NAMES)
        model_name = os.path.join(output, base + '.pth')
        torch.save(model.state_dict(), model_name)
        shutil.copyfile(class_names, out_classname)
        onnx_model_name = os.path.join(output, base + '.onnx')
        export_onnx_model(num_classes, model_name, onnx_model_name)
    else:
        pass

    return is_success

if __name__ == "__main__":
    num_epochs = 10
    dataset_path = "./dataset/voc_mask"
    output = "./output/ssd"
    train_result = start_train(num_epochs, dataset_path, output)
    if train_result:
        print("********** Training successfully **********")
    else:
        print("********** Training fail **********")
