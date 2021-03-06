# pip3 install efficientnet-pytorch
from efficientnet_pytorch import EfficientNet
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import torch.utils.model_zoo as model_zoo
import torchvision

logger = logging.getLogger('model_arch')


# NORM_LAYER = nn.BatchNorm2d() # num_features (output)
# NORM_LAYER = nn.GroupNorm() # num_groups, num_channels


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine does not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        x = self.conv(x)
        return x


def get_mesh(batch_size,
             shape_x,
             shape_y,
             device,
             ):
    mg_x, mg_y = np.meshgrid(np.linspace(0, 1, shape_y), np.linspace(0, 1, shape_x))
    mg_x = np.tile(mg_x[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mg_y = np.tile(mg_y[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mesh = torch.cat([torch.tensor(mg_x).to(device), torch.tensor(mg_y).to(device)], 1)
    return mesh


class MyUNet(nn.Module
             ):
    '''Mixture of previous classes'''

    def __init__(self,
                 n_classes,
                 device,
                 params,
                 ):
        # save params
        self.params = params
        self.device = device
        super(MyUNet, self).__init__()

        # setup layers
        self.mp = nn.MaxPool2d(2)
        if not self.params['model']['flag_use_dummy_model']:
            self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
            self.conv0 = double_conv(5, 64)
            self.conv1 = double_conv(64, 128)
            self.conv2 = double_conv(128, 512)
            self.conv3 = double_conv(512, 1024)
            self.up1 = up(1282 + 1024, 512)  # feats = 1282, x4 = 1024
            self.up2 = up(512 + 512, 256)  # x3 = 512
            if self.params['model']['factor_downsample'] == 4:
                self.up3 = up(256 + 128 + 1, 256)  # x2 = 128, mask=1
            self.outc = nn.Conv2d(256, n_classes, 1)
        else:
            logger.warning("!!! CAUTION: USING DUMMY MODEL !!!")
            logger.warning("!!! CAUTION: USING DUMMY MODEL !!!")
            logger.warning("!!! CAUTION: USING DUMMY MODEL !!!")
            # replicate base model with dummys: 5x maxpool and 1280 resulting channels
            self.base_model = nn.Sequential(
                nn.MaxPool2d(2),
                nn.MaxPool2d(2),
                nn.MaxPool2d(2),
                nn.MaxPool2d(2),
                nn.MaxPool2d(2),
                nn.Conv2d(3, 1280, 3, padding=1),
            )
            self.conv0 = double_conv(5, 1)
            self.conv1 = double_conv(1, 1)
            self.conv2 = double_conv(1, 1)
            self.conv3 = double_conv(1, 1)
            self.up1 = up(1282 + 1, 1)  # feats = 1282, x4 = 1024
            self.up2 = up(1 + 1, 1)  # x3 = 512
            if self.params['model']['factor_downsample'] == 4:
                self.up3 = up(1 + 1, 1)  # x2 = 128
            self.outc = nn.Conv2d(1, n_classes, 1)

    def forward(self, input):
        # extract image and gray-scale mask from input
        x = input[:, 0:3, :, :]
        mask = input[:, 3:, :, :]
        mask_resized = self.mp(self.mp(mask))

        # simply perform 4x double convolution + max-pooling
        batch_size = x.shape[0]
        mesh1 = get_mesh(batch_size, x.shape[2], x.shape[3], self.device)
        x0 = torch.cat([x, mesh1], 1)
        x1 = self.mp(self.conv0(x0))
        x2 = self.mp(self.conv1(x1))
        x3 = self.mp(self.conv2(x2))
        x4 = self.mp(self.conv3(x3))

        # in parallel, extract features using base model on center image (because rest is padding)
        img_width = x.shape[3]
        x_center = x[:, :, :, img_width // 8: -img_width // 8]
        if not self.params['model']['flag_use_dummy_model']:
            feats = self.base_model.extract_features(x_center)
        else:
            feats = self.base_model(x_center)

        bg = torch.zeros([feats.shape[0], feats.shape[1], feats.shape[2], feats.shape[3] // 8]).to(self.device)
        feats = torch.cat([bg, feats, bg], 3)
        mesh2 = get_mesh(batch_size, feats.shape[2], feats.shape[3], self.device)
        feats = torch.cat([feats, mesh2], 1)  # add positional info via mesh

        x = self.up1(feats, x4)  # upsample feat, concat result with x4 and conv
        x = self.up2(x, x3)  # upsample x, concat result with x3 and conv
        if self.params['model']['factor_downsample'] == 4:
            x2_mask = torch.cat([x2, mask_resized], 1)
            x = self.up3(x, x2_mask)  # upsample x, concat result with x2 and conv
        x = self.outc(x)
        return x


if __name__ == '__main__':
    # Gets the GPU if there is one, otherwise the cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # define model and test inference with dummy data
    params = {'model': {'factor_downsample': 4,
                        'flag_use_dummy_model': 0,
                        },
              }
    width = 1536  # 1536  # 1024
    height = 512  # 512  # 320
    model = MyUNet(8, device, params).to(device)
    size_batch = 3
    img_batch = torch.randn((size_batch, 4, height, width))
    mat_pred = model(img_batch.to(device))
    print(mat_pred)
    print(mat_pred.shape)
    print('=== Finished')
