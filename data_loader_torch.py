import numpy as np
import torch
import torch.utils.data
import data_loader
import cv2
import math
import matplotlib.pyplot as plt
import albumentations
import scripts.flip_image_hor
from tqdm import tqdm


def reduce_saturation(img, sat_shift_range=(-0.5, 0)):
    sat_shift = np.random.uniform(sat_shift_range[0], sat_shift_range[1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(img)
    sat_new = np.clip(cv2.add(sat, sat_shift), 0, 1.0)
    img = cv2.merge((hue, sat_new, val)).astype(img.dtype)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img


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
                 params,
                 flag_load_label=True,
                 flag_augment=True,
                 ):
        self.dataset = dataset
        self.flag_load_label = flag_load_label
        self.flag_augment = flag_augment
        self.factor_downsample = params['model']['factor_downsample']
        self.model_input_height = params['model']['input_height']
        self.model_input_width = params['model']['input_width']
        self.params = params

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx_item):
        # load item
        if torch.is_tensor(idx_item):
            idx_item = idx_item.tolist()
        id = self.dataset.list_ids[idx_item]
        item = self.dataset.load_item(id,
                                      flag_load_car=self.flag_load_label,
                                      flag_load_mask=self.params['datasets']['flag_use_mask'],
                                      )

        # preprocess image
        img = self.preprocess_img(item.img)

        # Convert car labels to matrix
        if self.flag_load_label:
            mat = self.convert_item_to_mat(item)
        else:
            mat = np.zeros((1, 1, 8))

        # perform augmentation
        if self.flag_augment:
            img, mat = self.augment_img(img, mat, idx_item)

        # convert matrices to desired torch format
        img = np.rollaxis(img, 2, 0)
        mask = mat[:, :, 0]
        regr = mat[:, :, 1:]
        regr = np.rollaxis(regr, 2, 0)

        return [img, mask, regr]

    def preprocess_img(self, img):
        img = img[img.shape[0] // 2:]  # only use bottom half of image
        bg = np.ones_like(img) * img.mean(1, keepdims=True).astype(img.dtype)
        bg = bg[:, :img.shape[1] // 6]  # add 1/6 padding to both sides -> new dims: [50%, 133%]
        img = np.concatenate([bg, img, bg], 1)
        img = cv2.resize(img, (self.model_input_width, self.model_input_height))
        img = img.astype('float32') / 255.0
        return img

    def augment_img(self, img, mat, idx_item=None):
        """ img already in float32 BGR format, not uint8
        """

        # horizontal flip
        p_flip = np.random.uniform()  # in [0,1)
        if p_flip > 0:  # 0.33:
            uv_cx = np.array([1686.2379, 0])
            IMG_SHAPE = (2710, 3384, 3)  # img.shape = h,w,c
            uv_cx_new = self.convert_uv_to_uv_preprocessed(uv_cx, IMG_SHAPE)
            cx_mat = uv_cx_new[0]
            cx_img = cx_mat * self.factor_downsample
            img_flipped = scripts.flip_image_hor.flip_hor_at_u(img, cx_img)
            mat_flipped = scripts.flip_image_hor.flip_hor_at_u(mat, cx_mat)
            mat_flipped[:, :, 4] *= -1  # x
            mat_flipped[:, :, 2] *= -1  # sin(yaw)
            mat_flipped[:, :, 3] *= -1  # roll
        else:
            img_flipped = img
            mat_flipped = mat

        # grayish - change HSV values
        p_sat = np.random.uniform()  # in [0,1)
        if p_sat > 0.33:
            img_desat = reduce_saturation(img_flipped, sat_shift_range=(-0.15, 0))
        else:
            img_desat = img_flipped

        # gamma change
        aug1 = albumentations.RandomGamma(gamma_limit=(80, 120),
                                          p=0.33,
                                          )

        # gaussian noise
        aug2 = albumentations.MultiplicativeNoise(multiplier=(0.85, 1.15),
                                                  elementwise=True,
                                                  per_channel=True,
                                                  p=0.33,
                                                  )

        # apply all augmentations to image
        aug_tot = albumentations.Compose([aug1, aug2], p=1)
        img_augmented = aug_tot(image=img_desat)['image']

        # for debugging purposes
        if False:
            fig, ax = plt.subplots(3, 2, figsize=(9, 6))
            ax[0][0].imshow(img[:, :, ::-1])
            ax[0][1].imshow(img_augmented[:, :, ::-1])
            ax[1][0].imshow(mat[:, :, 0])
            ax[1][1].imshow(mat_flipped[:, :, 0])
            ax[2][0].imshow(mat[:, :, 4])  # x
            ax[2][1].imshow(mat_flipped[:, :, 4])
            # fig.tight_layout()
            plt.show()
            fig.savefig('plots_aug/{:05d}.png'.format(idx_item))

        return img_augmented, mat_flipped

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
        num_cars = len(item.cars)
        for car in item.cars:
            uv_center = car.get_uv_center()
            uv_new = self.convert_uv_to_uv_preprocessed(uv_center, item.img.shape)
            uv_new = np.round(uv_new).astype(int)  # round floats to int
            u, v = uv_new[0], uv_new[1]
            if 0 <= u and u < mat.shape[1]:
                if 0 <= v and v < mat.shape[0]:
                    mat[v, u, 0] = 1  # confidence
                    mat[v, u, 1] = math.cos(car.yaw)
                    mat[v, u, 2] = math.sin(car.yaw)
                    mat[v, u, 3] = convert_roll_to_roll_new(car.roll)
                    mat[v, u, 4] = car.x / 100.
                    mat[v, u, 5] = car.y / 100.
                    mat[v, u, 6] = car.pitch
                    mat[v, u, 7] = car.z / 100.

        # in case of focal loss usage, create heatmap for each car
        if self.params['train']['loss']['flag_focal_loss']:
            mat[:, :, 0] = 0  # reset confidence values to 0
            height = mat.shape[0]
            width = mat.shape[1]
            us, vs = np.meshgrid(np.arange(width), np.arange(height))
            for car in item.cars:
                uv_center = car.get_uv_center()
                uv_new = self.convert_uv_to_uv_preprocessed(uv_center, item.img.shape)
                uv_new = np.round(uv_new).astype(int)  # round floats to int
                u, v = uv_new[0], uv_new[1]
                # choose sigma s.t. sigma = 5px at 10m and 8x downsample (40x128)
                sigma = 5 * 10 / car.z * 8 / self.factor_downsample
                heatmap = np.exp(- ((us - u) ** 2 + (vs - v) ** 2) / (2.0 * sigma))
                mat[:, :, 0] += heatmap
        # set values less than 1E-12 to 0 (probably not necessary, but easier)
        mask_low = mat[:, :, 0] < 1E-12
        mat[:, :, 0][mask_low] = 0

        return mat

    def convert_mat_to_item(self, mat):
        distance_min = 2 * 8 / self.factor_downsample  # usually 2, but increases to 4

        def calc_angle_from_sin_cos(angle_sin, angle_cos):
            checksum = np.sqrt(angle_sin ** 2 + angle_cos ** 2)
            angle_sin = angle_sin / checksum
            angle_cos = angle_cos / checksum
            pitch = np.arccos(angle_cos) * np.sign(angle_sin)
            return pitch

        # extract uv coords, where confidence > 0
        vs, us = np.where(mat[:, :, 0] > 0)
        logits = mat[vs, us, 0]

        # clear points which are close together.
        idx_sorted = np.argsort(logits)[::-1]
        for idx1 in idx_sorted:
            v1, u1 = vs[idx1], us[idx1]
            for idx2 in idx_sorted:
                if idx2 <= idx1:
                    continue  # already evaluated other way around
                else:
                    v2, u2 = vs[idx2], us[idx2]
                    distance = (u1 - u2) ** 2 + (v1 - v2) ** 2
                    if distance < distance_min ** 2:
                        logits[idx2] = -1

        # create cars
        item = data_loader.DataItem()
        item.cars = []
        for idx in idx_sorted:
            if logits[idx] > 0:
                v, u = vs[idx], us[idx]
                yaw_cos = mat[v, u, 1]
                yaw_sin = mat[v, u, 2]
                yaw = calc_angle_from_sin_cos(yaw_sin, yaw_cos)
                car = data_loader.Car(mat[v, u, 4] * 100,
                                      mat[v, u, 5] * 100,
                                      mat[v, u, 7] * 100,
                                      yaw,
                                      mat[v, u, 6],
                                      convert_roll_new_to_roll(mat[v, u, 3]),
                                      mat[v, u, 0],  # abusing id for logit
                                      u=u,
                                      v=v,
                                      )
                item.cars.append(car)

        return item

    def convert_uv_to_uv_preprocessed(self, uv, img_org_shape):
        ''' Problem: Preprocessing image changes uv coordinates. What are new uv-coords?
        '''
        # take into account top cropping and side padding
        u, v = uv[0], uv[1]
        height, width, nchannels = img_org_shape
        v -= height // 2  # crop top
        u += width // 6  # add padding to both sides
        height_new = height // 2
        width_new = width * (1 + 2 / 6)

        # take into account resizing of original image to match model input size
        u = u / width_new * self.model_input_width
        v = v / height_new * self.model_input_height

        # take into account downsampling of network
        u /= self.factor_downsample
        v /= self.factor_downsample

        # return
        uv = np.array([u, v])
        return uv


if __name__ == '__main__':
    params = {'model': {'factor_downsample': 4,
                        'input_height': 512,
                        'input_width': 1536,
                        },
              'train': {'loss': {'flag_focal_loss': 1}},
              'datasets': {'flag_use_mask': 1},
              }
    dataset = data_loader.DataSet(path_csv='../data/train.csv',
                                  path_folder_images='../data/train_images',
                                  path_folder_masks='../data/train_masks',
                                  )
    dataset_torch = DataSetTorch(dataset, params, flag_augment=True)
    num_items = len(dataset_torch)
    for idx_item in tqdm(range(num_items)):
        [img, mask, regr] = dataset_torch[idx_item]

        # reverse rolling backwards
        img = np.rollaxis(img, 0, 3)
        regr = np.rollaxis(regr, 0, 3)
        print(img.shape)
        print(regr.shape)

        # plot example
        if False:
            fig, ax = plt.subplots(3, 1, figsize=(10, 10))
            ax[0].imshow(img[:, :, ::-1])
            ax[1].imshow(mask)
            ax[2].imshow(regr[:, :, 0])
            fig.tight_layout()
            plt.show()
            if idx_item % 5 == 0:
                dummy = 0
            # fig.savefig("output/plot.png")

    print("=== Finished")
