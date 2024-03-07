#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loader

@author: arman
"""

import json
import multiprocessing
import os
from collections import OrderedDict

import imageio.v2 as imageio
import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import early_stopping, ModelCheckpoint
from matplotlib import pyplot as plt
from torch import optim, tensor
from torch.utils.data import Dataset, DataLoader, dataloader
from torchmetrics import PrecisionRecallCurve
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import (FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights,
                                                      FasterRCNN_ResNet50_FPN_V2_Weights)
from torchvision.ops import box_area, box_iou
from torchvision.transforms import v2 as transforms
from torchvision.tv_tensors import BoundingBoxes
from torchvision.utils import draw_bounding_boxes

# Input Parameters (CHANGE)
train_dir, train_json = 'train', 'train.json'
val_dir, val_json = 'val', 'val.json'
test_dir, test_json = 'test', 'test.json'
num_classes: int = 6  # Output classes of objects to be predicted; Either 3 or 6
base_model = 'resnet2'
# Other
labels_metrics_all_json = 'labels_metrics_all.json'

# Model Parameters
iou_threshold = 0.3
mean, std = [0.485], [0.229]
image_target_dims = [512, 512]  # Square is better!
scheduled_lr = False
batch_size = 1  # 7 for resnet, 1 for resnet2
lr = 1e-5
num_workers = multiprocessing.cpu_count()

default_root_dir = './'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# *** Bugs ***
# - Have to turn off GPU to run manual metrics calculations section

torch.set_float32_matmul_precision('medium')


class DentalDataset(Dataset):
    def __init__(self, images_dir, annotations_file, images_ext='.jpg', augmentation=False, num_classes=num_classes):
        with open(annotations_file) as f:
            annotations = json.load(f)

        # Annotation json contains several entries for each image (one for every box),
        # we make it so each image has one entry containing all boxes.
        image_annotations = OrderedDict()
        for annotation in annotations['annotation']:
            image_id = annotation.pop('image_id')

            if image_id not in image_annotations:
                # Can't append to a non-existing list...
                image_annotations[image_id] = []
            image_annotations[image_id].append(annotation)

        # bbox: [x1, y1, y2, x2]
        # boxes: [bbox1, bbox2, ...]
        # Expected model targets for each input: [boxes, labels]
        for key in image_annotations.keys():
            boxes = []
            labels = []
            for annotation in image_annotations[key]:
                boxes.append(annotation['bbox'])
                labels.append(annotation['category_id'])
            image_annotations[key] = {'boxes': tensor(boxes, dtype=torch.float),
                                      'labels': tensor(labels, dtype=torch.int64)}

        self.image_annotations = image_annotations
        self.image_ids = list(image_annotations.keys())
        self.images_dir = images_dir
        self.images_ext = images_ext
        self.augmentation = augmentation
        self.num_classes = num_classes

    def __len__(self):
        return len(self.image_annotations.keys())

    def __getitem__(self, index):
        # Fetch image
        image_id = self.image_ids[index]
        image = imageio.imread(os.path.join(self.images_dir,
                                            image_id + self.images_ext))
        target = self.image_annotations[image_id]

        # Preprocessing
        image, target = self.transform(image, target)

        # Classes: 3 or 6; For better performance, sometimes need to choose 3 class, merging
        if self.num_classes == 3:
            new_labels = []
            for label in target['labels']:
                if label <= 3:
                    label = 1
                elif label == 4:
                    label = 2
                elif label > 4:
                    label = 3
                new_labels.append(label)
            target['labels'] = torch.tensor(new_labels)

        return image, target

    def get_image_by_id(self, image_id):
        image = imageio.imread(os.path.join(self.images_dir,
                                            image_id + self.images_ext))
        target = self.image_annotations[image_id]

        # Preprocessing
        image, target = self.transform(image, target)

        # Classes: 3 or 6; For better performance, sometimes need to choose 3 class, merging
        new_labels = []
        for label in target['labels']:
            if label <= 3:
                label = 1
            elif label == 4:
                label = 2
            elif label > 4:
                label = 3
            new_labels.append(label)
        target['labels'] = torch.tensor(new_labels)

        return image, target

    def transform(self, image, target):
        # Augmentation
        if self.augmentation:
            # normalize commented out, unnecessary as these models do it internally
            transform_compose = transforms.Compose([
                # transforms.RandomRotation(degrees=5),
                # transforms.RandomAdjustSharpness(sharpness_factor=2),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
                transforms.RandomResizedCrop(size=image_target_dims, scale=(0.8, 1.0), antialias=True),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.Resize(image_target_dims, antialias=True),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            transform_compose = transforms.Compose([
                transforms.Resize(image_target_dims, antialias=True),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32),
                transforms.Normalize(mean=mean, std=std)
            ])

        canvas_size = image.shape
        # Adding color channel for gray image
        image = torch.from_numpy(image).unsqueeze(0)
        # Now image is in shape (1, H, W)
        # BoundingBoxes object is **necessary** for v2 transforms to work on bbox
        boxes = BoundingBoxes(target['boxes'], format='XYXY', canvas_size=canvas_size)
        # For now, not passing labels, only image and boxes (testing)
        image, boxes = transform_compose(image, boxes)

        # Ignore boxes with zero area. After transform, some small boxes have zero width/height.
        mask = box_area(boxes) != 0
        target['boxes'] = boxes[mask]
        target['labels'] = target['labels'][mask]

        return image, target


# Function later needed for dataloader
# Default collate of DataLoader created extra dimension in my batch targets,
# so I created a custom collate
def custom_collate(batch):
    images = [item[0] for item in batch]
    images = dataloader.default_collate(images)

    # Not applying default collate to this part, the problematic part
    targets = []
    for item in batch:
        target = {'boxes': item[1]['boxes'], 'labels': item[1]['labels']}
        targets.append(target)

    return images, targets


# Dataset Train/Val/Test
train_data = DentalDataset(train_dir, train_json, augmentation=True)
val_data = DentalDataset(val_dir, val_json)
test_data = DentalDataset(test_dir, test_json)

# DataLoader
train_loader = DataLoader(train_data, collate_fn=custom_collate, shuffle=True,
                          num_workers=num_workers, batch_size=batch_size)
val_loader = DataLoader(val_data, collate_fn=custom_collate, shuffle=False,
                        num_workers=num_workers, batch_size=batch_size)
test_loader = DataLoader(test_data, collate_fn=custom_collate, shuffle=False,
                         num_workers=num_workers, batch_size=batch_size)


class FasterRCNNLightning(pl.LightningModule):
    def __init__(self, num_classes=num_classes, lr=lr, batch_size=batch_size, base_model=base_model):
        super().__init__()
        self.num_classes = num_classes
        self.lr = lr  # learning rate
        self.batch_size = batch_size
        if base_model == 'resnet':
            self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        elif base_model == 'resnet2':
            self.model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features,
                                                               self.num_classes + 1)
        self.iou_thresholds = list(np.arange(0.1, 1.0, 0.1))  # Change for finer charts
        self.mAP = MeanAveragePrecision(iou_thresholds=[0.1, 0.2, 0.3])

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        # Model returns loss dict
        losses = sum(loss for loss in loss_dict.values())
        self.log('train_loss', losses, batch_size=self.batch_size, prog_bar=True)
        return losses

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        # Switch to train in order to get losses from model
        self.model.train()
        # Model returns loss dict
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log('val_loss', losses, batch_size=batch_size, prog_bar=True)
        # Back to eval?
        self.model.eval()
        return losses

    def test_step(self, batch, batch_idx):
        global cls_logits
        images, targets = batch
        outputs = self(images)
        # Test metrics
        self.mAP.update(outputs, targets)

    def on_test_end(self) -> None:
        print(self.mAP.compute())

    def forward(self, images):
        return self.model(images)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)

        if scheduled_lr:
            stepping_batches = self.trainer.estimated_stepping_batches
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=stepping_batches)
            return [optimizer], [scheduler]
        else:
            return [optimizer]


def visualize(image, targets=None, preds=None, threshold=None):
    image = image * std[0] + mean[0]  # Un-normalize
    image = torch.as_tensor(image, dtype=torch.uint8)
    if image.ndim < 3:
        image = image.unsqueeze(0)

    if targets:
        boxes, labels = targets['boxes'], targets['labels']
        labels = [str(x.numpy()) for x in labels]
        targets_to_show = draw_bounding_boxes(image, boxes=boxes, labels=labels)

    if preds:
        boxes, labels = preds['boxes'], preds['labels']
        if threshold:
            mask = preds['scores'] > threshold
            boxes, labels = boxes[mask], labels[mask]
            labels = [str(x.numpy()) for x in labels]  # To string, for plot labels
            preds_to_show = draw_bounding_boxes(image, boxes=boxes, labels=labels)
        else:
            labels = [str(x.numpy()) for x in labels]
            preds_to_show = draw_bounding_boxes(image, boxes=boxes, labels=labels)

    if targets and preds:
        # Show image with targets labeled
        targets_to_show = torch.as_tensor(targets_to_show, dtype=torch.float32)
        targets_to_show = transforms.ToPILImage()(targets_to_show)
        plt.subplot(1, 2, 1)
        plt.ion()
        plt.imshow(targets_to_show, cmap='gray')
        plt.axis('off')
        plt.title('Expert')
        plt.show()
        # Show image with predictions
        preds_to_show = torch.as_tensor(preds_to_show, dtype=torch.float32)
        preds_to_show = transforms.ToPILImage()(preds_to_show)
        plt.subplot(1, 2, 2)
        plt.ion()
        plt.imshow(preds_to_show, cmap='gray')
        plt.axis('off')
        plt.title('AI Prediction')
        plt.show()
    elif targets:
        # Show image with targets labeled
        targets_to_show = torch.as_tensor(targets_to_show, dtype=torch.float32)
        targets_to_show = transforms.ToPILImage()(targets_to_show)
        plt.ion()
        plt.imshow(targets_to_show, cmap='gray')
        plt.axis('off')
        plt.title('Expert')
        plt.show()
    elif preds:
        # Show image with predictions
        preds_to_show = torch.as_tensor(preds_to_show, dtype=torch.float32)
        preds_to_show = transforms.ToPILImage()(preds_to_show)
        plt.ion()
        plt.imshow(preds_to_show, cmap='gray')
        plt.axis('off')
        plt.title('AI Prediction')
        plt.show()
    else:
        image = torch.as_tensor(image, dtype=torch.float32)
        image = transforms.ToPILImage()(image)
        plt.ion()
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.title('Input')
        plt.show()


model = FasterRCNNLightning()

# Train the model
if input('Train the model? [y/N]: ').lower() == 'y':
    # Callbacks
    callbacks = [early_stopping.EarlyStopping(monitor='val_loss', mode='min', patience=10),
                 ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1,
                                 dirpath=os.path.join(default_root_dir, 'lightning_logs'), filename='{epoch}')]
    # Trainer
    trainer = pl.Trainer(max_epochs=200, default_root_dir=default_root_dir,
                         log_every_n_steps=40, callbacks=callbacks)
    # Fit
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
else:
    if input('Load model from checkpoint file? [Y/n]: ').lower() != 'n':
        ckpt_path = input('Enter path to model checkpoint: ')
        if ckpt_path:
            model = FasterRCNNLightning.load_from_checkpoint(ckpt_path)

# Test and metrics
if input('Do test? [y/N]: ').lower() == 'y':
    trainer = pl.Trainer()
    trainer.test(model=model, dataloaders=test_loader)

prompt_manual_metrics = input('Do manual metrics calculations? (S: show from previous JSON) [y/N/s]: ').lower()
if prompt_manual_metrics == 'y' or prompt_manual_metrics == 's':
    # Basic plot data variables
    labels = [str(label) for label in range(1, num_classes + 1)]  # Class names
    labels_metrics_all = []  # For each threshold
    recall_precision_plot = [[], []]  # (recall, precision)
    # Confidence thresholds ('score')
    thresholds = np.arange(0, 1, step=0.1)

    if prompt_manual_metrics == 'y':
        # Manual Test metrics
        # Set model to eval and freeze
        model.eval()
        model.freeze()
        # Predict and Calculate
        tp, fp, fn = 0, 0, 0
        outputs_all, targets_all = [], []
        for images, targets in test_loader:
            # Predict outputs
            with torch.no_grad():
                outputs = model(images.to(device))
            # Pack outputs and targets
            outputs_all.extend(outputs)
            targets_all.extend(targets)

        # Calculate metrics for each threshold
        for threshold in thresholds:
            # Labels
            labels_metrics = {'all': {'tn': 0}}
            for label in labels:
                labels_metrics[label] = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'R': 0, 'P': 0}  # R: Recall P: Precision

            for output, target in zip(outputs_all, targets_all):
                mask = output['scores'] > threshold
                output['boxes'] = output['boxes'][mask]
                output['labels'] = output['labels'][mask]
                output['scores'] = output['scores'][mask]
                # Match boxes based on IoUs
                gt_ious = box_iou(target['boxes'].to(device), output['boxes'].to(device))
                if gt_ious.shape[1] != 0 and gt_ious.shape[0] == len(target['labels']):
                    best_ious, ious_idx = gt_ious.max(1)
                    # Count tp, tn, fp, fn based on iou and label
                    for iou, iou_idx, target_iou_idx in zip(best_ious, ious_idx, range(len(ious_idx))):
                        label = str(target['labels'][target_iou_idx].numpy())
                        if iou > iou_threshold:
                            if output['labels'][iou_idx] == target['labels'][target_iou_idx]:
                                # True positive
                                labels_metrics[label]['tp'] += 1
                                # Other classes, true negative
                                labels_metrics['all']['tn'] += 1
                                labels_metrics[label]['tn'] -= 1
                            else:
                                # False positive
                                labels_metrics[label]['fp'] += 1
                        else:
                            # Missed
                            labels_metrics[label]['fn'] += 1
                elif gt_ious.shape[1] != 0 and gt_ious.shape(0) < len(target['labels']):
                    raise Exception('IoU result happened to have less returns than target labels length!')
                else:
                    # Missed everything
                    for label in target['labels']:
                        label = str(label.numpy())
                        labels_metrics[label]['fn'] += 1

            # Calculate F1 for current threshold for each class
            for label in labels:
                labels_metrics[label]['tn'] = labels_metrics['all']['tn'] + labels_metrics[label]['tn']
                tp, tn, fp, fn = (labels_metrics[label]['tp'], labels_metrics[label]['tn'], labels_metrics[label]['fp'],
                                  labels_metrics[label]['fn'])
                try:
                    precision = tp / (tp + fp)
                except ZeroDivisionError:
                    precision = None
                try:
                    recall = tp / (tp + fn)
                except ZeroDivisionError:
                    recall = None
                labels_metrics[label]['P'] = precision
                labels_metrics[label]['R'] = recall
            labels_metrics.pop('all')
            labels_metrics_all.append([threshold, labels_metrics])

        # Save to file for later use in case needed
        with open(labels_metrics_all_json, 'w') as fh:
            json.dump(labels_metrics_all, fh)

    # Load previous calculated JSON to save time
    if prompt_manual_metrics == 's':
        with open(labels_metrics_all_json) as fh:
            labels_metrics_all = json.load(fh)

    # Precision Recall plotting
    precision = {}
    recall = {}
    for label in labels:
        precision[label], recall[label] = [], []
        for threshold_idx in range(len(thresholds)):
            precision[label].append(labels_metrics_all[threshold_idx][1][label]['P'])
            recall[label].append(labels_metrics_all[threshold_idx][1][label]['R'])
            # Draw plot for label
        plt.plot(recall[label], precision[label], label=label)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    print('Precision: ', precision)
    print('Recall: ', recall)

# DEBUG
train_data = DentalDataset(train_dir, train_json, augmentation=True)
train_sample = train_data.get_image_by_id('515')
val_sample = DentalDataset(val_dir, val_json).get_image_by_id('6')
test_sample = DentalDataset(test_dir, test_json).get_image_by_id('26')
model.eval()
pred_train = model.to('cpu')(train_sample[0].unsqueeze(0))
pred_val = model.to('cpu')(val_sample[0].unsqueeze(0))
pred_test = model.to('cpu')(test_sample[0].unsqueeze(0))