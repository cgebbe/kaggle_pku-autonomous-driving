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


def main():
    # load params
    with open("params.yaml", 'r') as stream:
        params = yaml.safe_load(stream)

    # load model
    assert torch.cuda.is_available(), "cuda device is not available"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('Working on device={}'.format(device))
    model = model_architecture.MyUNet(8, device).to(device)
    path_weights = params['model']['path_weights']
    if path_weights:
        assert os.path.isfile(path_weights), "path_weights does not exist as a file"
        model.load_state_dict(torch.load(path_weights))

    # execute training or inference
    if params['mode'] == 'train':
        df_out = train.train(model,
                             device,
                             params,
                             )
        path_csv = os.path.join(params['path_folder_out'], 'train_history.csv')
        df_out.to_csv(path_csv, sep=';')
    elif params['mode'] == 'predict':
        df_out = predict.predict(model,
                                 device,
                                 params,
                                 )
        path_csv = os.path.join(params['path_folder_out'], 'predictions.csv')
        df_out.to_csv(path_csv, sep=',', index=False)

    logger.info("=== Finished")


if __name__ == '__main__':
    main()
