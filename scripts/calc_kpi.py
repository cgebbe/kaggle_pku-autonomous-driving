"""
directly from https://www.kaggle.com/its7171/metrics-evaluation-script
"""
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from math import sqrt, acos, pi, sin, cos
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from inspect import signature
import matplotlib.pyplot as plt
from multiprocessing import Pool

THRES_TR_LIST = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
THRES_RO_LIST = [50, 45, 40, 35, 30, 25, 20, 15, 10, 5]


def expand_df(df, PredictionStringCols):
    df = df.dropna().copy()
    df['NumCars'] = [int((x.count(' ') + 1) / 7) for x in df['PredictionString']]

    image_id_expanded = [item for item, count in zip(df['ImageId'], df['NumCars']) for i in range(count)]
    prediction_strings_expanded = df['PredictionString'].str.split(' ', expand=True).values.reshape(-1, 7).astype(float)
    prediction_strings_expanded = prediction_strings_expanded[~np.isnan(prediction_strings_expanded).all(axis=1)]
    df = pd.DataFrame(
        {
            'ImageId': image_id_expanded,
            PredictionStringCols[0]: prediction_strings_expanded[:, 0],
            PredictionStringCols[1]: prediction_strings_expanded[:, 1],
            PredictionStringCols[2]: prediction_strings_expanded[:, 2],
            PredictionStringCols[3]: prediction_strings_expanded[:, 3],
            PredictionStringCols[4]: prediction_strings_expanded[:, 4],
            PredictionStringCols[5]: prediction_strings_expanded[:, 5],
            PredictionStringCols[6]: prediction_strings_expanded[:, 6]
        })
    return df


def str2coords(s, names):
    coords = []
    for l in np.array(s.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, l.astype('float'))))
    return coords


def TranslationDistance(p, g, abs_dist=False):
    dx = p['x'] - g['x']
    dy = p['y'] - g['y']
    dz = p['z'] - g['z']
    diff0 = (g['x'] ** 2 + g['y'] ** 2 + g['z'] ** 2) ** 0.5
    diff1 = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
    if abs_dist:
        diff = diff1
    else:
        diff = diff1 / diff0
    return diff


def RotationDistance(p, g):
    true = [g['pitch'], g['yaw'], g['roll']]
    pred = [p['pitch'], p['yaw'], p['roll']]
    q1 = R.from_euler('xyz', true)
    q2 = R.from_euler('xyz', pred)
    diff = R.inv(q2) * q1
    W = np.clip(diff.as_quat()[-1], -1., 1.)

    # in the official metrics code:
    # https://www.kaggle.com/c/pku-autonomous-driving/overview/evaluation
    #   return Object3D.RadianToDegree( Math.Acos(diff.W) )
    # this code treat θ and θ+2π differntly.
    # So this should be fixed as follows.
    W = (acos(W) * 360) / pi
    if W > 180:
        W = 360 - W
    return W


def print_pr_curve(result_flg, scores, recall_total=1):
    average_precision = average_precision_score(result_flg, scores)
    precision, recall, _ = precision_recall_curve(result_flg, scores)
    recall *= recall_total
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.show()


def check_match(idx):
    keep_gt = False
    thre_tr_dist = THRES_TR_LIST[idx]
    thre_ro_dist = THRES_RO_LIST[idx]
    train_dict = {imgID: str2coords(s, names=['carid_or_score', 'pitch', 'yaw', 'roll', 'x', 'y', 'z']) for imgID, s in
                  zip(train_df['ImageId'], train_df['PredictionString'])}
    valid_dict = {imgID: str2coords(s, names=['pitch', 'yaw', 'roll', 'x', 'y', 'z', 'carid_or_score']) for imgID, s in
                  zip(pred_df['ImageId'], pred_df['PredictionString'])}
    result_flg = []  # 1 for TP, 0 for FP
    scores = []
    MAX_VAL = 10 ** 10
    for img_id in valid_dict:
        for pcar in sorted(valid_dict[img_id], key=lambda x: -x['carid_or_score']):
            # find nearest GT
            min_tr_dist = MAX_VAL
            min_idx = -1
            for idx, gcar in enumerate(train_dict[img_id]):
                tr_dist = TranslationDistance(pcar, gcar)
                if tr_dist < min_tr_dist:
                    min_tr_dist = tr_dist
                    min_ro_dist = RotationDistance(pcar, gcar)
                    min_idx = idx

            # set the result
            if min_tr_dist < thre_tr_dist and min_ro_dist < thre_ro_dist:
                if not keep_gt:
                    train_dict[img_id].pop(min_idx)
                result_flg.append(1)
            else:
                result_flg.append(0)
            scores.append(pcar['carid_or_score'])

    return result_flg, scores


if __name__ == '__main__':
    # params
    max_workers = 1
    path_csv_train = '../../data/train.csv'
    path_csv_pred = '../../output/20200110_img_larger_v3/predictions.csv'

    # load predicted and train dataset
    pred_df = pd.read_csv(path_csv_pred)
    pred_df = pred_df.fillna('')
    expanded_valid_df = expand_df(pred_df, ['pitch', 'yaw', 'roll', 'x', 'y', 'z', 'Score'])
    train_df = pd.read_csv(path_csv_train)
    train_df = train_df[train_df.ImageId.isin(pred_df.ImageId.unique())]
    expanded_train_df = expand_df(train_df, ['model_type', 'pitch', 'yaw', 'roll', 'x', 'y', 'z'])

    # calculate map
    n_gt = len(expanded_train_df)
    ap_list = []
    p = Pool(processes=max_workers)
    for result_flg, scores in p.imap(check_match, range(10)):
        if np.sum(result_flg) > 0: # fn o
            n_tp = np.sum(result_flg)
            recall = n_tp / n_gt
            if True: # see https://www.kaggle.com/its7171/metrics-evaluation-script#710936
                scores2 = np.random.rand(len(result_flg))
            ap = average_precision_score(result_flg, scores) * recall
            print_pr_curve(result_flg, scores, recall)
        else:
            ap = 0
        ap_list.append(ap)
    map = np.mean(ap_list)
    print('map={}'.format(map))
    print("=== Finished")
