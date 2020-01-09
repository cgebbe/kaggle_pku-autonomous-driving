# from tqdm.notebook import tqdm as tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import data_loader
import data_loader_torch
import torch
import torch.utils.data
import scipy
import scipy.optimize


def optimize_xyz(v_pred, u_pred, x0, y0, z0, params):
    def distance_fn(xyz):
        # constants
        IMG_SHAPE = (2710, 3384, 3)  # img.shape

        # calculate slope error
        x, y, z = xyz
        y_pred = 1.0392211185855782 + 0.05107277 * x + 0.16864302 * z  # from notebook
        slope_err = (y_pred - y) ** 2

        # calculate u,v
        cam_K = np.array([[2304.5479, 0, 1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)
        xyz = np.array([x, y, z])
        uv = data_loader.xyz2uv(xyz, cam_K)
        dataset_torch = data_loader_torch.DataSetTorch(None, params)
        uv_new = dataset_torch.convert_uv_to_uv_preprocessed(uv, IMG_SHAPE)
        u_new, v_new = uv_new[0], uv_new[1]

        # calc distance
        distance = (max(0.2, (v_new - v_pred) ** 2 + (u_new - u_pred) ** 2)
                    + max(0.4, slope_err))
        return distance

    res = scipy.optimize.minimize(distance_fn, [x0, y0, z0], method='Powell')
    x_new, y_new, z_new = res.x
    return x_new, y_new, z_new


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
    dataset_torch_test = data_loader_torch.DataSetTorch(dataset_test, params,
                                                        flag_load_label=False,
                                                        flag_augment=False,
                                                        )
    data_loader_test = torch.utils.data.DataLoader(dataset=dataset_torch_test,
                                                   batch_size=params['predict']['batch_size'],
                                                   shuffle=False,
                                                   num_workers=0,
                                                   )

    # perform predictions
    predictions = []
    model.eval()
    for idx_batch, (img, _, _) in enumerate(data_loader_test):
        print("{}/{} Predicting batch".format(idx_batch, len(data_loader_test)))
        if idx_batch > params['predict']['num_batches_max']:
            print("Ending early because of param num_batches_max={}".format(params['predict']['num_batches_max']))
            break

        # perform prediction
        with torch.no_grad():
            output = model(img.to(device))
        output = output.data.cpu().numpy()

        # extract cars as string from each element in batch
        num_elems_in_batch = output.shape[0]
        for idx_elem_in_batch in range(num_elems_in_batch):
            # get image in RGB format from tensor
            if params['predict']['flag_plot_mat'] or params['predict']['flag_plot_item']:
                assert params['predict']['batch_size'] == 1
                img_numpy = img.data.cpu().numpy()
                img_numpy = img_numpy[0, ...]
                img_numpy = np.rollaxis(img_numpy, 0, 3)  # reverse rolling backwards
                img_numpy = img_numpy[:, :, ::-1]  # BGR to RGB

            # get mat from output and plot
            mat = output[idx_elem_in_batch, ...]
            mat = np.rollaxis(mat, 0, 3)  # reverse rolling backwards
            if params['predict']['flag_plot_mat']:
                fig, ax = plt.subplots(3, 1, figsize=(12, 12))
                ax[0].imshow(img_numpy)
                ax[1].imshow(mat[:, :, 0])
                ax[2].imshow(mat[:, :, 1])
                fig.savefig('output/plot_mat.png')
                if False:
                    logits = mat[:, :, 0]
                    num_cars = np.sum(logits > 0)

            # convert mat to item and plot.
            item = dataset_torch_test.convert_mat_to_item(mat)
            if params['predict']['flag_optimize']:
                for idx_car, car in enumerate(item.cars):
                    x_new, y_new, z_new = optimize_xyz(car.v, car.u, car.x, car.y, car.z, params)
                    item.cars[idx_car].x = x_new
                    item.cars[idx_car].y = y_new
                    item.cars[idx_car].z = z_new

            if params['predict']['flag_plot_item']:
                id = dataset_test.df_cars.loc[idx_batch, 'ImageId']
                item.img = img_numpy
                item.mask = np.zeros((1, 1))
                fig, ax = item.plot()
                fig.suptitle('ImageID={}'.format(id))

                # save
                path_out = os.path.join(params['path_folder_out'],
                                        'pred_new',
                                        '{:05d}.png'.format(idx_batch),
                                        )
                os.makedirs(os.path.dirname(path_out), exist_ok=True)
                fig.savefig(path_out)
                plt.close()

            # extract prediction string from item
            string = item.get_cars_as_string(flag_submission=True)
            predictions.append(string)

    # predictions to csv
    df_out = dataset_torch_test.dataset.df_cars
    df_out.loc[0:len(predictions) - 1, 'PredictionString'] = predictions
    print(df_out.head())
    return df_out
