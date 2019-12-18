import numpy as np
import torch
import torch.utils.data
import data_loader
import cv2
import math


def convert_roll_to_roll_new(roll):
    '''
    Problem:
        - values are spread around -pi and pi in interval [-pi, +pi).
        - Thus, similar roll angles have very different values!
    Solution: Make values more similar by spreading them all around 0
    '''
    roll = roll % (2 * np.pi)  # -> spread around +pi in interval [0,2pi)
    roll = roll - np.pi  # -> spread around 0 in interval [-pi, +pi)
    return roll


def convert_roll_new_to_roll(roll):
    # incidentally backwards conversion works the same way !
    # -> spread around 0 and +2pi in interval [0,2pi)
    # -> spread around -pi and +pi in interval [-pi,pi)
    return convert_roll_to_roll_new(roll)


class DataSetTorch(torch.utils.data.Dataset):
    """ Wrapper around DataSet class.
    Notes:
        - includes preprocessing of items
        - can be directly used for training and inference
    """

    def __init__(self,
                 dataset,
                 model_input_height=512,
                 model_input_width=2048,
                 model_factor_downsample=8,
                 ):
        self.dataset = dataset
        self.factor_downsample = model_factor_downsample
        self.model_input_height = model_input_height
        self.model_input_width = model_input_width

    def __len__(self):
        return len(dataset)

    def __getitem__(self, idx_item):
        # load item
        if torch.is_tensor(idx_item):
            idx_item = idx_item.tolist()
        id = self.dataset.list_ids[idx_item]
        item = self.dataset.load_item(id)

        # preprocess image and label
        img = self.preprocess_img(item.img)
        mat = self.convert_item_to_mat(item)

        # output
        mask = mat[:, :, 0]
        regr = mat[:, :, 1:]
        return [img, mask, regr]

    def preprocess_img(self, img):
        # crop top and pad both sides horizontally
        height, width, nchannels = img.shape
        img_new_shape = (height // 2, width * 2, nchannels)
        img_new = np.zeros(img_new_shape, dtype=img.dtype)
        img_new[:, width // 2:width // 2 + width, :] = img[height // 2:, :, :]

        # resize and rescale features
        img_new = cv2.resize(img_new, (self.model_input_width, self.model_input_height))
        img_new = img_new.astype('float32') / 255.0
        return img_new

    def convert_item_to_mat(self, item):
        # create empty mat with 8 channels for
        # [confidence, x,y,z, yaw, pitch_sin, pitch_cos, roll]
        assert item.img is not None
        height, width, _ = item.img.shape
        mat_shape = (self.model_input_height // self.factor_downsample,
                     self.model_input_width // self.factor_downsample,
                     8)
        mat = np.zeros(mat_shape, dtype='float32')

        # fill in matrixes
        for car in item.cars:
            uv_center = car.get_uv_center()
            uv_new = self.convert_uv_to_uv_preprocessed(uv_center, item.img)
            u, v = uv_new[0], uv_new[1]
            if 0 <= u and u < self.model_input_width:
                if 0 <= v and v <= self.model_input_height:
                    mat[v, u, 0] = 1  # confidence
                    mat[v, u, 1] = car.x
                    mat[v, u, 2] = car.y
                    mat[v, u, 3] = car.z
                    mat[v, u, 4] = car.yaw
                    mat[v, u, 5] = math.sin(car.pitch)
                    mat[v, u, 6] = math.cos(car.pitch)
                    mat[v, u, 7] = convert_roll_to_roll_new(car.roll)
        return mat

    def convert_mat_to_item(self, mat, distance_min=2):
        def calc_pitch_from_sin_cos(pitch_sin, pitch_cos):
            checksum = np.sqrt(pitch_sin ** 2 + pitch_cos ** 2)
            pitch_sin = pitch_sin / checksum
            pitch_cos = pitch_cos / checksum
            pitch = np.arccos(pitch_cos) * np.sign(pitch_sin)
            return pitch

        # extract uv coords, where confidence > 0
        vs, us = np.where(mat[:, :, 0] > 0)
        confidences = mat[vs, us, 0]

        # clear points which are close together.
        idx_sorted = np.argsort(confidences)[::-1]
        for idx1 in idx_sorted:
            v1, u1 = vs[idx1], us[idx1]
            for idx2 in idx_sorted:
                if idx2 <= idx1:
                    continue  # already evaluated other way around
                else:
                    v2, u2 = vs[idx2], us[idx2]
                    distance = (u1 - u2) ** 2 + (v1 - v2) ** 2
                    if distance < distance_min ** 2:
                        confidences[idx2] = -1

        # create cars
        item = data_loader.DataItem()
        item.cars = []
        for idx in idx_sorted:
            if confidences[idx] > 0:
                v, u = vs[idx], us[idx]
                pitch_sin = mat[v, u, 5]
                pitch_cos = mat[v, u, 6]
                pitch = calc_pitch_from_sin_cos(pitch_sin, pitch_cos)
                car = data_loader.Car(mat[v, u, 1],
                                      mat[v, u, 2],
                                      mat[v, u, 3],
                                      mat[v, u, 4],
                                      pitch,
                                      convert_roll_new_to_roll(mat[v, u, 7]),
                                      mat[v, u, 0],  # abusing id for confidence
                                      )
                item.cars.append(car)

        return item

    def convert_uv_to_uv_preprocessed(self, uv, img_org):
        ''' Problem: Preprocessing image changes uv coordinates. What are new uv-coords?
        '''
        # take into account top cropping and side padding
        u, v = uv[0], uv[1]
        height, width, nchannels = img_org.shape
        v -= height // 2  # crop top
        u += width // 2  # add padding to both sides

        # take into account resizing of original image
        height_new = height // 2
        width_new = width * 2
        u = u / width_new * self.model_input_width
        v = v / height_new * self.model_input_height

        # take into account downsampling of network
        u /= self.factor_downsample
        v /= self.factor_downsample

        # round up to be a pixel value
        uv = np.array([u, v])
        uv = np.round(uv).astype(int)
        return uv


if __name__ == '__main__':
    dataset = data_loader.DataSet('../data/training')
    dataset_torch = DataSetTorch(dataset)
