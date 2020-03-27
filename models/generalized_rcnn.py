# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

from collections import OrderedDict
import torch
from torch import nn


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, backbone2, rpn, roi_heads, transform, is_double_backbone=True):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        if is_double_backbone:
            self.backbone = backbone
        else:
            self.backbone = None
        self.backbone2 = backbone2
        self.rpn = rpn
        self.roi_heads = roi_heads
        
        self.is_double_backbone = is_double_backbone
       
    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = None
        if self.is_double_backbone:
            features = self.backbone(images.tensors)

        features2 = self.backbone2(images.tensors)       

        if self.is_double_backbone:
            if isinstance(features, torch.Tensor):
                new_features = OrderedDict([(0, torch.cat([features, features2], 1))])
            else:
                new_features = OrderedDict([ (0, torch.cat([features[0], features2[0]], 1)), 
                (1, torch.cat([features[1], features2[1]], 1)), 
                (2, torch.cat([features[2], features2[2]], 1)), 
                (3, torch.cat([features[3], features2[3]], 1)), 
                ('pool', torch.cat([features['pool'], features2['pool']], 1))])
        else:
            if isinstance(features, torch.Tensor):
                new_features = OrderedDict([(0, features2)])
            else:
                new_features = features2
                
        proposals, proposal_losses = self.rpn(images, new_features, targets)
        detections, detector_losses = self.roi_heads(new_features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training:
            return losses

        return detections
