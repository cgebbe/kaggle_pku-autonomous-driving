import data_loader
import data_loader_torch
import torch.utils.data
# from tqdm.notebook import tqdm as tqdm
import torch.optim
import gc
import pandas as pd


def criterion(prediction, mask, regr, weight=0.4, size_average=True):
    # Binary mask loss
    pred_mask = torch.sigmoid(prediction[:, 0])
    # mask_loss = mask * (1 - pred_mask)**2 * torch.log(pred_mask + 1e-12) + (1 - mask) * pred_mask**2 * torch.log(1 - pred_mask + 1e-12)
    mask_loss = mask * torch.log(pred_mask + 1e-12) + (1 - mask) * torch.log(1 - pred_mask + 1e-12)
    mask_loss = -mask_loss.mean(0).sum()

    # Regression L1 loss
    pred_regr = prediction[:, 1:]
    regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
    regr_loss = regr_loss.mean(0)

    # Sum
    loss = weight * mask_loss + (1 - weight) * regr_loss
    if not size_average:
        loss *= prediction.shape[0]
    return loss, mask_loss, regr_loss


def evaluate(model,
             device,
             data_loader_valid,
             params,
             idx_epoch=0,
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

            if idx_epoch < params['train']['idx_epoch_switch']:
                loss, mask_loss, regr_loss = criterion(output, mask_batch, regr_batch, 1, size_average=False)
                valid_loss += loss.data
                valid_mask_loss += mask_loss.data
                valid_regr_loss += regr_loss.data
            else:
                loss, mask_loss, regr_loss = criterion(output, mask_batch, regr_batch, 0.5, size_average=False)
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
    dataset_torch_train = data_loader_torch.DataSetTorch(dataset_train)
    dataset_torch_valid = data_loader_torch.DataSetTorch(dataset_valid)
    data_loader_train = torch.utils.data.DataLoader(dataset=dataset_torch_train,
                                                    batch_size=params['train']['batch_size'],
                                                    shuffle=True,
                                                    num_workers=0,
                                                    )
    data_loader_valid = torch.utils.data.DataLoader(dataset=dataset_torch_valid,
                                                    batch_size=params['train']['batch_size_eval'],
                                                    shuffle=True,
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
            print("{}/{} Training batch".format(idx_batch, num_batches))
            img_batch = img_batch.to(device)
            mask_batch = mask_batch.to(device)
            regr_batch = regr_batch.to(device)

            optimizer.zero_grad()
            output = model(img_batch)
            if idx_epoch < params['train']['idx_epoch_switch']:
                loss, mask_loss, regr_loss = criterion(output, mask_batch, regr_batch, 1)
            else:
                loss, mask_loss, regr_loss = criterion(output, mask_batch, regr_batch, 0.5)
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
                                   idx_epoch,
                                   )
        for key, value in values_per_name.items():
            df_out.loc[idx_epoch, key] = value
        print(df_out)

        # evaluate(epoch, history) # commented out, because have not defined a valid dataset
        torch.save(model.state_dict(), 'model_{}.pth'.format(idx_epoch))
    return df_out
