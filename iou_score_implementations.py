import torch
import numpy as np
from collections import defaultdict


def multi_class_iou_numpy(y_true, y_pred, threshold = 0.5, epsilon=0.0001, cls_ids = [0,1]):
    """
    args:
        y_true: numpy.ndarray of shape (batch_size, num_classes, height, width)
        y_pred: numpy.ndarray of shape (batch_size, num_classes, height, width)
        threshold: threshold for prediction
        epsilon: small number to avoid division by zero
        cls_ids: list of class ids to compute IoU for

    returns:
        ious: defaultdict containing the IoU for each class and the mean IoU
    """

    ious = defaultdict(int)

    for cls_id in cls_ids:

        y_pred = y_pred > threshold
        y_pred = y_pred.astype(np.float32)

        tp = np.logical_and(y_true == cls_id, y_pred == cls_id)
        fp = np.logical_and(y_true != cls_id, y_pred == cls_id)
        fn = np.logical_and(y_true == cls_id, y_pred != cls_id)

        iou = (np.sum(tp) + epsilon) / (np.sum(tp) + np.sum(fp) + np.sum(fn) + epsilon)
        ious[f"class {cls_id} IoU"] = iou
    
    ious["mean IoU"] = np.mean(list(ious.values()))
    return ious

def multi_class_iou_torch(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001, cls_ids = [0,1]):
    """
    args:
        y_true: torch.tensor of shape (batch_size, num_classes, height, width)
        y_pred: torch.tensor of shape (batch_size, num_classes, height, width)
        thr: threshold for prediction
        dim: dimensions to sum over
        epsilon: small number to avoid division by zero
        cls_ids: list of class ids to compute IoU for
    
    returns:
        ious: defaultdict containing the IoU for each class and the mean IoU
    """

    ious = defaultdict(int)

    for cls_id in cls_ids:

        y_pred = y_pred > thr
        y_pred = y_pred.type(torch.float32)

        tp = torch.logical_and(y_true == cls_id, y_pred == cls_id)
        fp = torch.logical_and(y_true != cls_id, y_pred == cls_id)
        fn = torch.logical_and(y_true == cls_id, y_pred != cls_id)

        iou = (torch.sum(tp) + epsilon) / (torch.sum(tp) + torch.sum(fp) + torch.sum(fn) + epsilon)

        ious[f"class {cls_id} IoU"] = iou.item()
    
    ious["mean IoU"] = np.mean(list(ious.values()))
    return ious 

y_true = np.array([[[[0, 0, 0, 1, 1, 1],
                        [0, 0, 0, 1, 1, 1],
                        [0, 0, 0, 1, 1, 1],
                        [0, 0, 0, 1, 1, 1],
                        [0, 0, 0, 1, 1, 1],
                        [0, 0, 0, 1, 1, 1]]]])

y_pred = np.array([[[[0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1],
                        [0, 0, 0, 1, 1, 1],
                        [0, 0, 0, 1, 1, 1]]]])

iou_np = multi_class_iou_numpy(y_true, y_pred)
iou_torch = multi_class_iou_torch(torch.from_numpy(y_true), torch.from_numpy(y_pred))
