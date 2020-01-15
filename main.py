import yaml
import torch
import logging
import os
import argparse

import model_architecture
import train
import predict

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--path_yaml',
                    default='params.yaml',
                    )
args = parser.parse_args()

# setup logger - TODO: right format and save log in output folder
logger = logging.getLogger('main')


def main():
    # load params
    with open(args.path_yaml, 'r') as stream:
        params = yaml.safe_load(stream)

    # create output folder
    os.makedirs(params['path_folder_out'], exist_ok=True)

    # choose device for loading
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("WARNING !!! Using CPU ")
        device = torch.device("cpu")
    logger.info('Working on device={}'.format(device))

    # load model
    num_classes = 8
    model = model_architecture.MyUNet(num_classes, device, params).to(device)
    path_weights = params['model']['path_weights']
    if path_weights:
        assert os.path.isfile(path_weights), "path_weights does not exist as a file"
        model.load_state_dict(torch.load(path_weights, map_location=device))

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

    logger.info("=== Finished")


if __name__ == '__main__':
    main()
