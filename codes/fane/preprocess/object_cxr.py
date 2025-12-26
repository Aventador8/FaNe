import ipdb
import math
import numpy as np
import pandas as pd
from codes.fane.constants import *
import pickle
from shapely.geometry import LineString
from shapely.algorithms.polylabel import polylabel
from shapely.ops import unary_union
from sklearn.model_selection import train_test_split

np.random.seed(0)


OBJECT_SEP = ';'
ANNOTATION_SEP = ' '


def rectangle_box(anno):
    x = []
    y = []

    anno = anno[2:]
    anno = anno.split(ANNOTATION_SEP)
    for i in range(len(anno)):
        if i % 2 == 0:
            x.append(int(anno[i]))
        else:
            y.append(int(anno[i]))

    xmin = min(x)
    xmax = max(x)
    ymin = min(y)
    ymax = max(y)
    w = xmax - xmin
    h = ymax - ymin
    box = [xmin, ymin, w, h]
    return box

def polylabel_box(anno):
    polygon = anno.split(ANNOTATION_SEP)[1:]
    polygon = list(map(int, polygon))
    p = LineString(np.asarray(polygon + polygon[:2]).reshape(-1, 2))
    c = polylabel(p.buffer(100), tolerance=10)
    box = np.asarray(p.bounds).reshape(-1, 2)  # [[x_min, y_min], [x_max, y_max]]
    cxy = np.array([c.x, c.y])  # 转换为 numpy 数组 [x_center, y_center]
    # 计算宽度和高度
    wh = np.abs(box[1] - box[0])  # [x_max - x_min, y_max - y_min]
    # 构造边界框 [x_min, y_min, w, h]
    box = [box[0][0], box[0][1], wh[0], wh[1]]
    return box


# source: https://github.com/xuyuan/xsd/blob/master/data/object_cxr_to_coco.ipynb
def annotation_to_bbox(annotation):
    bbox = []

    if not annotation:
        return bbox

    annotation_list = annotation.split(OBJECT_SEP)
    for anno in annotation_list:
        if anno[0] in ('0', '1'):
            box = rectangle_box(anno)
        elif anno[0] == '2':
            box = polylabel_box(anno)
        else:
            raise RuntimeError(anno[0])
        bbox.append(box)
    return bbox


def save_pkl(df, pkl_path):
    filenames, bboxs_list = [], []
    for row in df.itertuples():
        filenames.append(row.image_name)
        if row.annotation != row.annotation:
            bboxs_list.append(np.zeros((1, 4)))
        else:
            bboxs = annotation_to_bbox(row.annotation)
            bboxs_list.append(bboxs)

    filenames = np.array(filenames)
    bboxs_list = np.array(bboxs_list,dtype=object)
    with open(pkl_path, "wb") as f:
        pickle.dump([filenames, bboxs_list], f)


def main():
    ori_train_df = pd.read_csv(OBJ_ORIGINAL_TRAIN_CSV)
    # ori_train_df.dropna(subset=["annotation"], inplace=True)
    # ori_train_df.reset_index(drop=True, inplace=True)

    train_df, val_df = train_test_split(
        ori_train_df, test_size=0.1, random_state=0)

    save_pkl(train_df, OBJ_TRAIN_PKL)
    save_pkl(val_df, OBJ_VALID_PKL)

    test_df = pd.read_csv(OBJ_ORIGINAL_DEV_CSV)
    # test_df.dropna(subset=["annotation"], inplace=True)
    # test_df.reset_index(drop=True, inplace=True)
    save_pkl(test_df, OBJ_TEST_PKL)


if __name__ == "__main__":
    main()