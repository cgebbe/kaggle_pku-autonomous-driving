import os
import data_loader
import data_loader_torch
import torch.utils.data
# from tqdm.notebook import tqdm as tqdm
import torch.optim
import gc
import pandas as pd


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt (batch x c x h x w)

      taken from https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/losses.py
    '''
    alpha = 2
    beta = 4

    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, beta)
    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
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
        pred = torch.sigmoid(prediction[:, 0])
        mask_loss = _neg_loss(pred, gt)
    else:
        # Binary mask loss
        pred_mask = torch.sigmoid(prediction[:, 0])
        mask_loss = mask * torch.log(pred_mask + 1e-12) + (1 - mask) * torch.log(1 - pred_mask + 1e-12)
        mask_loss = -mask_loss.mean(0).sum()

    # Regression L1 loss
    pred_regr = prediction[:, 1:]
    regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
    regr_loss = regr_loss.mean(0)

    # Sum
    loss = mask_loss + regr_loss
    if not params['flag_size_average']:
        loss *= prediction.shape[0]
    return loss, mask_loss, regr_loss


def evaluate(model,
             device,
             data_loader_valid,
             params,
             ):
    # perform evaluation
    model.eval()
    valid_loss = 0
    valid_mask_loss = 0
    valid_regr_loss = 0
    with torch.no_grad():
        num_batches = len(data_loader_valid)
        for idx_batch, (img_batch, mask_batch, regr_batch) in enumerate(data_loader_valid):
            print("{}/{} Evaluating batch".format(idx_batch, num_batches))
            img_batch = img_batch.to(device)
            mask_batch = mask_batch.to(device)
            regr_batch = regr_batch.to(device)
            output = model(img_batch)


            loss, mask_loss, regr_loss = calc_loss(
                output, mask_batch, regr_batch, params['train']['loss'],)
            valid_loss += loss.data
            valid_mask_loss += mask_loss.data
            valid_regr_loss += regr_loss.data

    # calculate average loss
    valid_loss /= len(data_loader_valid.dataset)
    valid_mask_loss /= len(data_loader_valid.dataset)
    valid_regr_loss /= len(data_loader_valid.dataset)

    # return output
    values_per_name = {}
    values_per_name["valid_loss"] = valid_loss.cpu().numpy()
    values_per_name["valid_loss_mask"] = valid_mask_loss.cpu().numpy()
    values_per_name["valid_loss_regr"] = valid_regr_loss.cpu().numpy()
    return values_per_name


def train(model,
          device,
          params,
          ):
    # Create data generators - they will produce batches
    dataset_train = data_loader.DataSet(
        path_csv=params['datasets']['train']['path_csv'],
        path_folder_images=params['datasets']['train']['path_folder_images'],
        path_folder_masks=params['datasets']['train']['path_folder_masks'],
    )
    dataset_valid = data_loader.DataSet(
        path_csv=params['datasets']['valid']['path_csv'],
        path_folder_images=params['datasets']['valid']['path_folder_images'],
        path_folder_masks=params['datasets']['valid']['path_folder_masks'],
    )

    # wrap around datasets
    dataset_torch_train = data_loader_torch.DataSetTorch(
        dataset_train, params)
    dataset_torch_valid = data_loader_torch.DataSetTorch(
        dataset_valid, params, flag_augment=False)
    data_loader_train = torch.utils.data.DataLoader(dataset=dataset_torch_train,
                                                    batch_size=params['train']['batch_size'],
                                                    shuffle=True,
                                                    num_workers=0,
                                                    )
    data_loader_valid = torch.utils.data.DataLoader(dataset=dataset_torch_valid,
                                                    batch_size=params['train']['batch_size_eval'],
                                                    shuffle=False,
                                                    num_workers=0,
                                                    )

    # define optimizer
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=params['train']['learning_rate'],
                                  )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(params['train']['num_epochs'], 10) * len(data_loader_train) // 3,
        gamma=0.1,
    )

    # perform training
    df_out = pd.DataFrame()
    for idx_epoch in range(params['train']['num_epochs']):
        torch.cuda.empty_cache()
        gc.collect()
        model.train()
        loss_epoch = 0
        loss_mask_epoch = 0
        loss_regr_epoch = 0

        # data_loader_train_tqdm = tqdm(data_loader_train)
        num_batches = len(data_loader_train)
        for idx_batch, (img_batch, mask_batch, regr_batch) in enumerate(data_loader_train):
            print("epoch={}, {}/{} Training batch".format(idx_epoch, idx_batch, num_batches))
            img_batch = img_batch.to(device)
            mask_batch = mask_batch.to(device)
            regr_batch = regr_batch.to(device)

            optimizer.zero_grad()
            output = model(img_batch)
            loss, mask_loss, regr_loss = calc_loss(output,
                                                   mask_batch,
                                                   regr_batch,
                                                   params['train']['loss'],
                                                   )
            loss_epoch += loss.data
            loss_mask_epoch += mask_loss.data
            loss_regr_epoch += regr_loss.data

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        # collect epoch results
        loss_epoch /= len(data_loader_train.dataset)
        loss_mask_epoch /= len(data_loader_train.dataset)
        loss_regr_epoch /= len(data_loader_train.dataset)
        df_out.loc[idx_epoch, 'train_loss'] = loss_epoch.cpu().numpy()
        df_out.loc[idx_epoch, 'train_loss_mask'] = loss_mask_epoch.cpu().numpy()
        df_out.loc[idx_epoch, 'train_loss_regr'] = loss_regr_epoch.cpu().numpy()

        # evaluate valid dataset
        values_per_name = evaluate(model,
                                   device,
                                   data_loader_valid,
                                   params,
                                   )
        for key, value in values_per_name.items():
            df_out.loc[idx_epoch, key] = value

        # save history
        path_csv = os.path.join(params['path_folder_out'],
                                'train_history_{}.csv'.format(idx_epoch),
                                )
        df_out.to_csv(path_csv, sep=';')
        print(df_out)

        # save model weights
        path_weights = os.path.join(params['path_folder_out'],
                                    'model_{}.pth'.format(idx_epoch),
                                    )
        torch.save(model.state_dict(), path_weights)
    return df_out
