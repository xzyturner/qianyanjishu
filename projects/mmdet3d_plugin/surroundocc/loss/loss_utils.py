import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
def multiscale_supervision(gt_occ, ratio, gt_shape):
    '''
    change ground truth shape as (B, W, H, Z) for each level supervision
    '''
    gt_occa = gt_occ.cpu().numpy()
    # gt_occ[0][:, :3] -= 1
    # gt_occ[:, 4] -= 1
    gt = torch.zeros([gt_shape[0], gt_shape[2], gt_shape[3], gt_shape[4]]).to(gt_occ.device).type(torch.float)
    for i in range(gt.shape[0]):
        coords = gt_occ[i][:, :3].type(torch.long) // ratio
        gt[i, coords[:, 0], coords[:, 1], coords[:, 2]] =  gt_occ[i][:, 3]

    return gt

def multiscale_supervision2d(gt_occ2d, ratio, gt_shape):
    '''
    change ground truth shape as (B, W, H, Z) for each level supervision
    '''
    gt_occa = gt_occ2d.cpu().numpy()
    # gt_occ[0][:, :3] -= 1
    # gt_occ[:, 4] -= 1
    gt = torch.zeros([gt_shape[0], gt_shape[2], gt_shape[3]]).to(gt_occ2d.device).type(torch.float)
    for i in range(gt.shape[0]):
        coords = gt_occ2d[i][:, :2].type(torch.long) // ratio
        gt[i, coords[:, 0],  coords[:, 1]] =  gt_occ2d[i][:, 2]

    return gt

def make2dgt(point_cloud):
    # 获取所有的 x, y, z, label
    point_cloud = point_cloud.squeeze(0)
    x_values = point_cloud[:, 0]
    y_values = point_cloud[:, 1]
    z_values = point_cloud[:, 2]
    labels = point_cloud[:, 3]

    # 创建一个新的点云列表，存储降维后的 (x, z, label)
    new_point_cloud = []

    # 获取所有不同的 (x, z) 对
    unique_xz_pairs = np.unique(point_cloud[:, [0, 2]], axis=0)

    # 遍历每个 (x, z) 对
    for xz in unique_xz_pairs:
        # 找到所有 xz 对应的点
        mask = (x_values == xz[0]) & (z_values == xz[1])
        xz_points = point_cloud[mask]

        # 在这些点中，找到 y 值最大的点
        max_y_point = xz_points[np.argmax(xz_points[:, 1])]

        # 将该点的 (x, z, label) 添加到新的点云中
        new_point_cloud.append([max_y_point[0], max_y_point[2], max_y_point[3]])
    new_point_cloud = torch.tensor(new_point_cloud).unsqueeze(0)
    return new_point_cloud

def geo_scal_loss(pred, ssc_target, semantic=True):

    # Get softmax probabilities
    if semantic:
        pred = F.softmax(pred, dim=1)

        # Compute empty and nonempty probabilities
        empty_probs = pred[:, 0, :, :, :]
    else:
        empty_probs = 1 - torch.sigmoid(pred)
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != 255
    nonempty_target = ssc_target != 0
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / nonempty_target.sum()
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
        + F.binary_cross_entropy(recall, torch.ones_like(recall))
        + F.binary_cross_entropy(spec, torch.ones_like(spec))
    )

def geo_scal_loss2d(pred, ssc_target, semantic=True):

    # Get softmax probabilities
    if semantic:
        pred = F.softmax(pred, dim=1)

        # Compute empty and nonempty probabilities
        empty_probs = pred[:, 0, :,  :]
    else:
        empty_probs = 1 - torch.sigmoid(pred)
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != 255
    nonempty_target = ssc_target != 0
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / nonempty_target.sum()
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
        + F.binary_cross_entropy(recall, torch.ones_like(recall))
        + F.binary_cross_entropy(spec, torch.ones_like(spec))
    )

def sem_scal_loss(pred, ssc_target):
    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    # print("ssc_target2:")
    # print(ssc_target.shape)
    # print(ssc_target)
    mask = ssc_target != 255
    # print("mask2:")
    # print(mask.shape)
    # print(mask)
    n_classes = pred.shape[1]
    for i in range(0, n_classes):

        # Get probability of class i
        p = pred[:, i, :, :, :]
        # print("p1:")
        # print(p.shape)
        # print(p)
        # Remove unknown voxels
        target_ori = ssc_target
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        completion_target_ori = torch.ones_like(target_ori).float()
        completion_target_ori[target_ori != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0
            if torch.sum(p) > 0:
                precision = nominator / (torch.sum(p))
                loss_precision = F.binary_cross_entropy(
                    precision, torch.ones_like(precision)
                )
                loss_class += loss_precision
            if torch.sum(completion_target) > 0:
                recall = nominator / (torch.sum(completion_target))
                loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                loss_class += loss_recall
            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                    torch.sum(1 - completion_target)
                )
                loss_specificity = F.binary_cross_entropy(
                    specificity, torch.ones_like(specificity)
                )
                loss_class += loss_specificity
            loss += loss_class
    return loss / count

def sem_scal_loss2d(pred, ssc_target):
    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    # print("ssc_target2:")
    # print(ssc_target.shape)
    # print(ssc_target)
    mask = ssc_target != 255
    # print("mask2:")
    # print(mask.shape)
    # print(mask)
    n_classes = pred.shape[1]
    for i in range(0, n_classes):

        # Get probability of class i
        p = pred[:, i, :,  :]
        # print("p1:")
        # print(p.shape)
        # print(p)
        # Remove unknown voxels
        target_ori = ssc_target
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        completion_target_ori = torch.ones_like(target_ori).float()
        completion_target_ori[target_ori != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0
            if torch.sum(p) > 0:
                precision = nominator / (torch.sum(p))
                loss_precision = F.binary_cross_entropy(
                    precision, torch.ones_like(precision)
                )
                loss_class += loss_precision
            if torch.sum(completion_target) > 0:
                recall = nominator / (torch.sum(completion_target))
                loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                loss_class += loss_recall
            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                    torch.sum(1 - completion_target)
                )
                loss_specificity = F.binary_cross_entropy(
                    specificity, torch.ones_like(specificity)
                )
                loss_class += loss_specificity
            loss += loss_class
    return loss / count