import os
import numpy as np
import matplotlib.pyplot as plt
import data_loader
import data_loader_torch
import torch.utils.data
from tqdm import tqdm
# from tqdm.notebook import tqdm as tqdm
import torch.optim
import gc
import pandas as pd


def _neg_loss(pred_org, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt (batch x c x h x w)

      taken from https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/losses.py
    '''
    alpha = 2
    beta = 4

    # convert pred to [0,1] range. Prevent exact 0 or 1, because would yield nan
    pred = torch.sigmoid(pred_org)
    eps = 1E-10
    pred = torch.clamp(pred, eps, 1 - eps)

    # separate into pos and neg loss
    ind_gt_eq1 = gt.eq(1).float()
    ind_gt_lt1 = gt.lt(1).float()
    num_pos = ind_gt_eq1.float().sum()

    # calc pos and neg loss
    neg_weights = torch.pow(1 - gt, beta)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * ind_gt_eq1
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * ind_gt_lt1
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    # calc total loss
    loss = 0
    if num_pos > 0:
        loss = loss - (pos_loss + neg_loss) / num_pos
    else:
        loss = loss - neg_loss
    return loss


def calc_loss(prediction,
              mask,
              regr,
              params,
              ):
    # for mask loss, use either focal loss or simpler binary mask loss
    if params['flag_focal_loss']:
        # focal loss, see https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/losses.py
        gt = mask
        pred = prediction[:, 0, :, :]
        loss_mask = _neg_loss(pred, gt)
        weight_mask = 0.5 / 20.0 * 15  # so that mask_loss more or less equal to regr_loss

        # for debugging only
        if False:
            idx_elem_in_batch = 0
            mask_np = mask.data.cpu().numpy()
            pred_np = pred.data.cpu().numpy()
            fig, ax = plt.subplots(2, 1)
            ax[0].imshow(mask_np[idx_elem_in_batch, ...])
            ax[1].imshow(pred_np[idx_elem_in_batch, ...])
            plt.show()
            fig.savefig('plot.png')
    else:
        # Binary mask loss
        pred_mask = torch.sigmoid(prediction[:, 0])
        loss_mask = mask * torch.log(pred_mask + 1e-12) + (1 - mask) * torch.log(1 - pred_mask + 1e-12)
        loss_mask = -loss_mask.mean(0).sum()
        weight_mask = 0.5 / 20.0  # so that mask_loss more or less equal to regr_loss
    loss_mask *= weight_mask

    # Regression L1 loss
    if params['flag_focal_loss']:
        mask_binary = mask.ge(1).float()
        if False:  # for debug purposes
            mask_np = mask.data.cpu().numpy()
            mask_binary_np = mask_binary.data.cpu().numpy()
            num_cars = np.sum(mask_binary_np)
            fig, ax = plt.subplots(2, 1, figsize=(15, 15))
            ax[0].imshow(mask_np[0, ...])
            ax[1].imshow(mask_binary_np[0, ...])
            fig.savefig('plot_mask.png')
    else:
        mask_binary = mask
    pred_regr = prediction[:, 1:]
    loss_regr = torch.abs(pred_regr - regr).sum(1) * mask_binary
    num_cars = mask_binary.sum(1).sum(1)
    loss_regr = (loss_regr.sum(1).sum(1) / num_cars).mean(0)

    # total loss
    loss_tot = loss_mask + loss_regr
    if not params['flag_size_average']:
        loss_tot *= prediction.shape[0]

    # calculate total loss
    loss_per_name = dict()
    loss_per_name['mask'] = loss_mask
    loss_per_name['regr'] = loss_regr
    loss_per_name['tot'] = loss_tot
    return loss_per_name


def evaluate(model,
             device,
             params,
             ):
    # define dataset
    dataset = data_loader.DataSet(
        path_csv=params['datasets']['valid']['path_csv'],
        path_folder_images=params['datasets']['valid']['path_folder_images'],
        path_folder_masks=params['datasets']['valid']['path_folder_masks'],
    )
    dataset_torch = data_loader_torch.DataSetTorch(dataset, params, flag_augment=False)
    dataset_loader = torch.utils.data.DataLoader(dataset=dataset_torch,
                                                 batch_size=params['train']['batch_size_eval'],
                                                 shuffle=False,
                                                 num_workers=0,
                                                 )

    # set model to eval (affects e.g. dropout layers) and disable unnecessary grad computation
    model.eval()
    torch.set_grad_enabled(False)
    torch.cuda.empty_cache()  # empty cuda cache to prevent memory errors
    gc.collect()  # empty unreferenced objects at the end. Not sure whether better at the end?

    # calculate loss for whole dataset
    num_batches = len(dataset_loader)
    loss_per_name = dict()
    print("Evaluating")
    for img_batch, mask_batch, regr_batch in tqdm(dataset_loader):
        # perform inference and calculate loss
        output = model(img_batch.to(device))
        batch_loss_per_name = calc_loss(output,
                                        mask_batch.to(device),
                                        regr_batch.to(device),
                                        params['train']['loss'],
                                        )
        for name, batch_loss in batch_loss_per_name.items():
            if name not in loss_per_name:
                loss_per_name[name] = 0
            loss_per_name[name] += batch_loss.data

    # calculate average
    for name, loss in loss_per_name.items():
        loss_per_name[name] = loss.cpu().numpy() / len(dataset_loader)
    len_dataset = len(dataset_loader.dataset)  # check difference
    len_dataset2 = len(dataset_loader)
    return loss_per_name


def train(model,
          device,
          params,
          ):
    # define training dataset
    dataset = data_loader.DataSet(
        path_csv=params['datasets']['train']['path_csv'],
        path_folder_images=params['datasets']['train']['path_folder_images'],
        path_folder_masks=params['datasets']['train']['path_folder_masks'],
    )
    dataset_torch = data_loader_torch.DataSetTorch(
        dataset,
        params,
        flag_load_label=True,
        flag_augment=params['train']['use_augmentation'],
    )
    dataset_loader = torch.utils.data.DataLoader(
        dataset=dataset_torch,
        batch_size=params['train']['batch_size'],
        shuffle=True,
        num_workers=0,
    )

    # define optimizer and decrease learning rate by 0.1 every 3 epochs
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=params['train']['learning_rate']['initial'],
                                  )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=params['train']['learning_rate']['num_epochs_const'],
        gamma=params['train']['learning_rate']['factor_decrease'],
    )

    # for each epoch...
    df_out = pd.DataFrame()
    for idx_epoch in range(params['train']['num_epochs']):
        print("Training epoch {}".format(idx_epoch))

        # set model to train (affects e.g. dropout layers) and disable unnecessary grad computation
        model.train()
        torch.set_grad_enabled(True)
        torch.cuda.empty_cache()  # empty cuda cache to prevent memory errors
        gc.collect()  # empty unreferenced objects at the end. Not sure whether better at the end?

        # calculate loss for whole dataset
        num_batches = len(dataset_loader)
        loss_per_name = dict()
        dataset_tqdm = tqdm(dataset_loader)
        for img_batch, mask_batch, regr_batch in dataset_tqdm:

            # perform inference and calculate loss
            output = model(img_batch.to(device))
            batch_loss_per_name = calc_loss(output,
                                            mask_batch.to(device),
                                            regr_batch.to(device),
                                            params['train']['loss'],
                                            )
            for name, batch_loss in batch_loss_per_name.items():
                if name not in loss_per_name:
                    loss_per_name[name] = 0
                loss_per_name[name] += batch_loss.data

            # change tqdm progress bar description
            description = "loss: "
            for name, batch_loss in batch_loss_per_name.items():
                description += "{}={:.3f} ".format(name, batch_loss.data.cpu().numpy())
            dataset_tqdm.set_description(description)

            # perform optimization
            batch_loss_per_name['tot'].backward()  # computes x.grad += dloss/dx for all parameters x
            optimizer.step()  # updates values x += -lr * x.grad
            optimizer.zero_grad()  # set x.grad = 0, for next iteration

        # step learning rate after each epoch (not after each batch)
        lr_scheduler.step()

        # calculate average and store results
        for name in loss_per_name.keys():
            loss_per_name[name] = loss_per_name[name].cpu().numpy() / len(dataset_loader)
            df_out.loc[idx_epoch, 'loss_' + name] = loss_per_name[name]
        values_per_name = evaluate(model, device, params)
        for key, value in values_per_name.items():
            df_out.loc[idx_epoch, 'valid_' + key] = value

        # save history
        path_csv = os.path.join(params['path_folder_out'],
                                'train_history_{}.csv'.format(idx_epoch),
                                )
        os.makedirs(os.path.dirname(path_csv), exist_ok=True)
        df_out.to_csv(path_csv, sep=';')
        print(df_out)

        # save model weights
        path_weights = os.path.join(params['path_folder_out'],
                                    'model_{}.pth'.format(idx_epoch),
                                    )
        torch.save(model.state_dict(), path_weights)
    return df_out
