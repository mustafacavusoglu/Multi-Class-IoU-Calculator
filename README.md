Certainly! Here's the GitHub README.md file content translated into English:

# Multi-Class Intersection over Union (IoU) Calculation

This repository contains Python code for calculating Intersection over Union (IoU) for multi-class segmentation problems. Both NumPy and PyTorch versions are provided.

## Contents
1. [Code Description](#code-description)
2. [Usage](#usage)
3. [Examples](#examples)

## Code Description

### multi_class_iou_numpy Function
This function calculates multi-class IoU using NumPy. Here are the important parameters and their functions:

- `y_true`: A NumPy array of real labels with shape (batch_size, num_classes, height, width).
- `y_pred`: A NumPy array of predicted labels with shape (batch_size, num_classes, height, width).
- `threshold`: Threshold used for binary classification of predictions (default value: 0.5).
- `epsilon`: A small number used to prevent division by zero (default value: 0.0001).
- `cls_ids`: A list containing class identifiers for which IoU is computed (e.g., `[0, 1]`).

The function computes IoU separately for each class and stores the results in a dictionary. It also calculates the mean IoU.

### multi_class_iou_torch Function
This function calculates multi-class IoU using PyTorch. Here are the important parameters and their functions:

- `y_true`: A PyTorch tensor of real labels with shape (batch_size, num_classes, height, width).
- `y_pred`: A PyTorch tensor of predicted labels with shape (batch_size, num_classes, height, width).
- `thr`: Threshold used for binary classification of predictions (default value: 0.5).
- `dim`: A tuple specifying the dimensions to sum over during computation (default value: (2, 3), which sums over height and width dimensions).
- `epsilon`: A small number used to prevent division by zero (default value: 0.001).
- `cls_ids`: A list containing class identifiers for which IoU is computed (e.g., `[0, 1]`).

The function computes IoU separately for each class and stores the results in a dictionary. It also calculates the mean IoU.

## Usage

This code can be used to evaluate the accuracy of predictions in multi-class segmentation problems. Here are usage examples:

```python
import torch
import numpy as np

# Example input data
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

# Calculate IoU with NumPy
iou_np = multi_class_iou_numpy(y_true, y_pred)

iou_np_class_0 = multi_class_iou_numpy(y_true, y_pred, cls_ids=[0])

# Calculate IoU with PyTorch
iou_torch = multi_class_iou_torch(torch.from_numpy(y_true), torch.from_numpy(y_pred))

iou_torch_class_0 = multi_class_iou_torch(torch.from_numpy(y_true), torch.from_numpy(y_pred), cls_ids=[0])
```

## Examples

This repository includes examples of using both NumPy and PyTorch to calculate IoU for multi-class segmentation problems. You can run these examples to see how the functions can be used in more detail.

---
