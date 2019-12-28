import yaml
import torch
import logging
import os

import model_architecture
import data_loader_torch
import data_loader
import train
import predict

logger = logging.getLogger('main')

with open("params.yaml", 'r') as stream:
    try:
        print(yaml.safe_load(stream))
    except yaml.YAMLError as exc:
        print(exc)


def main():
    # load params
    with open("params.yaml", 'r') as stream:
        params = yaml.safe_load(stream)

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('Working on device={}'.format(device))
    model = model_architecture.CentResnet(8, device).to(device)
    path_weights = params['model']['path_weights']
    if path_weights:
        assert os.path.isfile(path_weights), "path_weights does not exist as a file"
        model.load_state_dict(torch.load(path_weights))

    # load dataset
    dataset = data_loader.DataSet(path_csv=params['dataset']['path_csv'],
                                  path_folder_images=params['dataset']['path_folder_images'],
                                  path_folder_masks=params['dataset']['path_folder_masks'],
                                  )
    dataset_torch = data_loader_torch.DataSetTorch(dataset)

    # execute training or inference
    if params['mode'] == 'train':
        df_out = train.train_per_epoch(model,
                                       device,
                                       dataset_torch,
                                       params['train'],
                                       )
        df_out.to_csv('out_train.csv')
    elif params['mode'] == 'predict':
        df_out = predict.predict(model,
                                 device,
                                 dataset_torch,
                                 params['predict'],
                                 )
        df_out.to_csv('out_predict.csv')

    logger.info("=== Finished")


if __name__ == '__main__':
    main()
