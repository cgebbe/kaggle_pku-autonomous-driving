# from tqdm.notebook import tqdm as tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import data_loader
import data_loader_torch
import torch
import torch.utils.data


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
    dataset_torch_test = data_loader_torch.DataSetTorch(dataset_test)
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
            mat = output[idx_elem_in_batch, ...]
            mat = np.rollaxis(mat, 0, 3)  # reverse rolling backwards
            item = dataset_torch_test.convert_mat_to_item(mat)
            string = item.get_cars_as_string(flag_submission=True)
            predictions.append(string)

            # if visualize
            if params['predict']['flag_plot_item']:
                assert params['predict']['batch_size'] == 1
                id = dataset_test.df_cars.loc[idx_batch, 'ImageId']
                item_img = dataset_test.load_item(id, flag_load_mask=False, flag_load_car=False)
                item.img = item_img.img
                item.mask = np.zeros((1, 1))
                fig, ax = item.plot()
                fig.suptitle('ImageID={}'.format(id))

                # save
                path_out = os.path.join(params['path_folder_out'],
                                        'pred',
                                        '{:05d}.png'.format(idx_batch),
                                        )
                os.makedirs(os.path.dirname(path_out), exist_ok=True)
                fig.savefig(path_out)

    # predictions to csv
    df_out = dataset_torch_test.dataset.df_cars
    df_out.loc[0:len(predictions) - 1, 'PredictionString'] = predictions
    print(df_out.head())
    return df_out
