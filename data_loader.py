import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import logging
import math

logger = logging.getLogger('data_set')


# convert euler angle to rotation matrix
def euler_to_Rot(yaw, pitch, roll):
    Y = np.array([[math.cos(yaw), 0, math.sin(yaw)],
                  [0, 1, 0],
                  [-math.sin(yaw), 0, math.cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, math.cos(pitch), -math.sin(pitch)],
                  [0, math.sin(pitch), math.cos(pitch)]])
    R = np.array([[math.cos(roll), -math.sin(roll), 0],
                  [math.sin(roll), math.cos(roll), 0],
                  [0, 0, 1]])
    return np.dot(Y, np.dot(P, R))


def xyz2uv(xyz, K):
    if xyz.shape[0] == 4:
        xyz = xyz[0:3, ...]
    assert xyz.shape[0] == 3
    uvl = np.dot(K, xyz)
    uv = uvl[0:2] / uvl[2]
    return uv


class Car:
    """ xyz in camera coordinate system
    """

    def __init__(self, x, y, z, yaw, pitch, roll, id, u=None, v=None):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.roll = float(roll)
        self.id = id  # usually int, but abused as float logits value when predicting
        self.u = u
        self.v = v
        self.is_marked = 0

        # camera matrix K from camera_intrinsic.txt
        self.cam_K = np.array([[2304.5479, 0, 1686.2379],
                               [0, 2305.8757, 1354.9849],
                               [0, 0, 1]], dtype=np.float32)

    def get_uv_center(self):
        xyz_center = np.array([self.x, self.y, self.z]).T  # 3x1 vector
        uv_center = xyz2uv(xyz_center, self.cam_K)
        uv_center = np.round(uv_center).astype(np.int)
        return uv_center

    def plot(self, ax):
        # plot car center
        uv_center = self.get_uv_center()
        color_dot = 'red' if self.is_marked else 'green'
        ax.scatter(uv_center[0], uv_center[1], s=100, color=color_dot, alpha=0.5)

        # plot corner points
        x_dim = 1.02
        y_dim = 0.80
        z_dim = 2.31
        xyz_corners = np.array([[+x_dim, -y_dim, -z_dim, 1],
                                [+x_dim, -y_dim, +z_dim, 1],
                                [-x_dim, -y_dim, +z_dim, 1],
                                [-x_dim, -y_dim, -z_dim, 1],
                                [+x_dim, -y_dim, -z_dim, 1],
                                ]).T  # 4xN
        Rt = np.eye(4)
        Rt[0:3, 0:3] = euler_to_Rot(-self.yaw, -self.pitch, -self.roll).T
        Rt[0:3, 3] = np.array([self.x, self.y, self.z]).T  # 3x1 vector
        xyz_corners_rot = np.dot(Rt, xyz_corners)
        uv_corners = xyz2uv(xyz_corners_rot, self.cam_K)
        uv_corners = np.round(uv_corners).astype(np.int)
        ax.plot(uv_corners[0, :], uv_corners[1, :], color='red')


class DataItem:
    def __init__(self):
        self.img = None
        self.cars = None
        self.mask = None

    def set_cars_from_string(self, string):
        """ From instruction: string is concatenated list of values ['id', 'pitch', 'yaw', 'roll', 'x', 'y', 'z']
        Sidenotes:
            Documentation describes string with yaw and pitch interchanged, but wrong.
            xyz in camera coordinate system
        """
        # determine number of cars
        values = string.split(' ')
        assert len(values) % 7 == 0
        num_items_ = len(values) // 7

        # transform values into car object instances
        self.cars = []
        for idx_car in range(num_items_):
            car = Car(values[idx_car * 7 + 4],
                      values[idx_car * 7 + 5],
                      values[idx_car * 7 + 6],
                      values[idx_car * 7 + 2],  # yaw
                      values[idx_car * 7 + 1],  # pitch
                      values[idx_car * 7 + 3],  # roll
                      values[idx_car * 7 + 0],  # id
                      )
            self.cars.append(car)

    def get_cars_as_string(self, flag_submission=False):
        values = []
        for car in self.cars:
            if flag_submission:
                # car.id is abused to store logits
                confidence = 1 / (1 + np.exp(-car.id))
                assert (0 <= confidence and confidence <= 1), "confidence not in [0,1]"
                values.extend([car.pitch, car.yaw, car.roll, car.x, car.y, car.z, confidence])
                # values.extend([car.yaw, car.pitch, car.roll, car.x, car.y, car.z, confidence])
            else:
                values.extend([car.id, car.pitch, car.yaw, car.roll, car.x, car.y, car.z])
        values = [str(x) for x in values]
        string = ' '.join(values)
        return string

    def plot(self):
        # plot image and mask
        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(10, 10))
        ax.imshow(self.img[:, :, ::-1])

        # plot cars on top of image
        for car in self.cars:
            car.plot(ax)

        # show result
        fig.tight_layout()
        return fig, ax


class DataSet:
    def __init__(self,
                 path_csv,
                 path_folder_images,
                 path_folder_masks,
                 ):
        self.path_folder_images = path_folder_images
        self.path_folder_masks = path_folder_masks

        # parse csv file
        assert os.path.isfile(path_csv)
        self.df_cars = pd.read_csv(path_csv, sep=',')

        # remove erroneous images from list in training
        if 'train_train' in path_csv:
            ids_erroneous = ['ID_1a5a10365,'
                             'ID_4d238ae90',
                             'ID_408f58e9f',
                             'ID_bb1d991f6',
                             'ID_c44983aeb',
                             ]
            num_items_before = len(self.df_cars)
            for id in ids_erroneous:
                mask = self.df_cars.loc[:, 'ImageId'] == id
                assert np.sum(mask) in [0, 1]
                self.df_cars = self.df_cars.loc[np.invert(mask), :]
            num_items_after = len(self.df_cars)
            print("Deleted {} erroneous images".format(num_items_after - num_items_before))

        # determine id list from csv
        self.list_ids = list(self.df_cars.loc[:, 'ImageId'])

    def __len__(self):
        return len(self.list_ids)

    def load_item(self,
                  id,
                  flag_load_img=True,
                  flag_load_mask=False,
                  flag_load_car=True,
                  ):
        # construct empty item
        item = DataItem()

        # load image
        if flag_load_img:
            path_img = os.path.join(self.path_folder_images, id + '.jpg')
            item.img = cv2.imread(path_img)

        # load mask
        if flag_load_mask:
            path_mask = os.path.join(self.path_folder_masks, id + '.jpg')
            try:
                item.mask = cv2.imread(path_mask)
                mask_bool_per_channel = item.mask > 127
                mask_bool = np.all(mask_bool_per_channel, axis=-1)
                item.img[mask_bool] = [255, 0, 255]
            except:
                logger.debug('Mask not found for id={}'.format(id))

        # load car information
        if flag_load_car:
            mask_id = self.df_cars.loc[:, 'ImageId'] == id
            assert np.sum(mask_id) == 1
            cars_as_str = self.df_cars.loc[mask_id, 'PredictionString'].values[0]
            item.set_cars_from_string(cars_as_str)

        return item


if __name__ == '__main__':
    dataset = DataSet(path_csv='../data/train.csv',
                      path_folder_images='../data/train_images',
                      path_folder_masks='../data/train_masks',
                      )
    num_items = len(dataset)

    # plot distribution of roll angles
    if False:
        list_roll = []
        for idx_item, id in enumerate(dataset.list_ids):
            print("{}/{}".format(idx_item, num_items))
            item = dataset.load_item(id, flag_load_img=False, flag_load_mask=False)
            for car in item.cars:
                list_roll.append(car.roll)
        roll_min = np.min(list_roll)
        roll_max = np.max(list_roll)
        plt.hist(list_roll, bins=200)
        plt.show()

    # plot
    for idx_item, id in enumerate(dataset.list_ids):
        if idx_item > 10:
            continue
        item = dataset.load_item(id, flag_load_mask=True)
        fig, ax = item.plot()
        plt.show()
        # fig.savefig('output/plot_data_loader.png')

    print("=== Finished")
