import yaml
import torch
import logging
import os

import model_architecture
import train
import predict

logger = logging.getLogger('main')


def main():
    # load params
    with open("params.yaml", 'r') as stream:
        params = yaml.safe_load(stream)

    # TODO - setup logging

    # create output folder
    os.makedirs(params['path_folder_out'], exist_ok=True)

    # load model
    assert torch.cuda.is_available(), "cuda device is not available"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('Working on device={}'.format(device))
    num_classes = 8
    model = model_architecture.MyUNet(num_classes, device, params).to(device)
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
