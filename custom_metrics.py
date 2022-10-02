
import pandas as pd
import numpy as np


def get_iou(true_box, pred_box):
    ix1 = np.maximum(true_box[0], pred_box[0])
    iy1 = np.maximum(true_box[1], pred_box[1])
    ix2 = np.minimum(true_box[2], pred_box[2])
    iy2 = np.minimum(true_box[3], pred_box[3])

    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))

    area_of_intersection = i_height * i_width

    gt_height = true_box[3] - true_box[1] + 1
    gt_width = true_box[2] - true_box[0] + 1

    pd_height = pred_box[3] - pred_box[1] + 1
    pd_width = pred_box[2] - pred_box[0] + 1

    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection

    iou = area_of_intersection / area_of_union

    return iou
