# from tqdm.notebook import tqdm as tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import data_loader
import data_loader_torch
import torch
import torch.utils.data
import scipy
import scipy.optimize
from tqdm import tqdm


def optimize_xyz(v_pred, u_pred, x0, y0, z0, params):
    def calc_distance(xyz):
        # calculate fitting error
        x, y, z = xyz
        y_pred = 1.0392211185855782 + 0.05107277 * x + 0.16864302 * z  # from notebook
        dist_fit = (y_pred - y) ** 2

        # calculate u,v for reduced image
        IMG_SHAPE = (2710, 3384, 3)  # img.shape = h,w,c
        cam_K = np.array([[2304.5479, 0, 1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)
        xyz = np.array([x, y, z])
        uv = data_loader.xyz2uv(xyz, cam_K)
        dataset_torch = data_loader_torch.DataSetTorch(None, params)
        uv_new = dataset_torch.convert_uv_to_uv_preprocessed(uv, IMG_SHAPE)
        u_new, v_new = uv_new[0], uv_new[1]
        dist_uv = (v_new - v_pred) ** 2 + (u_new - u_pred) ** 2

        # total
        dist_tot = max(0.2, dist_uv) + max(0.4, dist_fit)
        # dist_tot = dist_uv + dist_fit
        return dist_tot

    # fit coordinates to u,v and y(x,z)
    res = scipy.optimize.minimize(calc_distance, [x0, y0, z0], method='Powell')
    x_new, y_new, z_new = res.x

    # check magnitude of change
    norm0 = np.linalg.norm([x0, y0, z0])
    # norm_new = np.linalg.norm([x_new, y_new, z_new])
    norm_diff = np.linalg.norm([x_new - x0, y_new - y0, z_new - z0])
    norm_diff_rel = norm_diff / norm0
    # print("\n norm_diff_rel={}".format(norm_diff_rel))

    # return either optimized or non optimized variables
    if norm_diff_rel > 0.33 or z_new < 0:
        is_marked = 1
        return x0, y0, z0, is_marked
    else:
        is_marked = 0
        return x_new, y_new, z_new, is_marked


def predict(model,
            device,
            params,
            ):
    # Create data generators - they will produce batches
    dataset_test = data_loader.DataSet(
        path_csv=params['datasets']['test']['path_csv'],
        path_folder_images=params['datasets']['test']['path_folder_images'],
        path_folder_masks=params['datasets']['test']['path_folder_masks'],
    )
    dataset_torch_test = data_loader_torch.DataSetTorch(
        dataset_test, params,
        flag_load_label=False,
        flag_augment=False,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset=dataset_torch_test,
        batch_size=params['predict']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,  # see https://pytorch.org/docs/stable/data.html
    )

    # perform predictions
    predictions = []
    model.eval()
    idx_batch = -1
    for img, mask, _, _ in tqdm(data_loader_test):
        idx_batch += 1
        if idx_batch > params['predict']['num_batches_max']:
            print("Ending early because of param num_batches_max={}".format(params['predict']['num_batches_max']))
            break

        # perform prediction
        with torch.no_grad():
            # concat img and mask and perform inference
            input = torch.cat([img, mask], 1)  # nbatch, nchannels, height, width
            output = model(input.to(device))
        output = output.data.cpu().numpy()

        # extract cars as string from each element in batch
        num_elems_in_batch = output.shape[0]
        for idx_elem_in_batch in range(num_elems_in_batch):
            idx_id = idx_batch * params['predict']['batch_size'] + idx_elem_in_batch
            id = dataset_test.list_ids[idx_id]

            # get mat from output and plot
            mat = output[idx_elem_in_batch, ...]
            mat = np.rollaxis(mat, 0, 3)  # reverse rolling backwards
            if params['predict']['flag_plot_mat']:
                # convert image to numpy
                img_numpy = img.data.cpu().numpy()
                img_numpy = img_numpy[idx_elem_in_batch, ...]
                img_numpy = np.rollaxis(img_numpy, 0, 3)  # reverse rolling backwards
                img_numpy = img_numpy[:, :, ::-1]  # BGR to RGB

                fig, ax = plt.subplots(2, 1, figsize=(10, 10))
                ax[0].imshow(img_numpy)
                ax_mask = ax[1].imshow(mat[:, :, 0], cmap='PiYG', vmin=-1, vmax=+1)
                fig.colorbar(ax_mask, ax=ax[1])
                if False:  # only use in case of multiple axes, here x,y,z
                    ax[2].imshow(mat[:, :, 4])
                    ax[3].imshow(mat[:, :, 5])
                    ax[4].imshow(mat[:, :, 7])
                for axi, label in zip(ax, ['img', 'mask']):  # , 'x', 'y', 'z']):
                    axi.set_ylabel(label)
                fig.suptitle('ImageID={}'.format(id))

                # save
                path_out = os.path.join(params['path_folder_out'],
                                        'pred_mat',
                                        '{:05d}.png'.format(idx_id),
                                        )
                os.makedirs(os.path.dirname(path_out), exist_ok=True)
                fig.savefig(path_out)
                plt.close()

            # convert mat to item and plot.
            item = dataset_torch_test.convert_mat_to_item(mat)
            if params['predict']['flag_optimize']:
                for idx_car, car in enumerate(item.cars):
                    x_new, y_new, z_new, is_marked = optimize_xyz(car.v, car.u, car.x, car.y, car.z, params)
                    item.cars[idx_car].x = x_new
                    item.cars[idx_car].y = y_new
                    item.cars[idx_car].z = z_new
                    item.cars[idx_car].is_marked = is_marked

            if params['predict']['flag_plot_item']:
                item_org = dataset_test.load_item(id)
                item.img = item_org.img
                item.mask = np.zeros((1, 1))
                fig, ax = item.plot()
                fig.suptitle('ImageID={}'.format(id))

                if idx_batch == 2:
                    num_cars = len(item.cars)
                    plt.show()

                # save
                path_out = os.path.join(params['path_folder_out'],
                                        'pred_item',
                                        '{:05d}.png'.format(idx_id),
                                        )
                os.makedirs(os.path.dirname(path_out), exist_ok=True)
                fig.savefig(path_out)
                plt.close()

            # extract prediction string from item
            string = item.get_cars_as_string(flag_submission=True)
            predictions.append(string)

    # predictions to csv
    df_out = pd.DataFrame()
    df_out['ImageId'] = dataset_test.list_ids
    df_out.loc[0:len(predictions) - 1, 'PredictionString'] = predictions
    print(df_out.head())
    path_csv = os.path.join(params['path_folder_out'], 'predictions.csv')
    df_out.to_csv(path_csv, sep=',', index=False)

    return df_out
