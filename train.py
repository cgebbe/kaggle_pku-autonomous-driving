import torch.utils.data
from tqdm.notebook import tqdm as tqdm
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
             dataset_torch,
             params,
             idx_epoch=0,
             ):
    # define dataset
    dev_loader = torch.utils.data.DataLoader(dataset=dataset_torch,
                                             batch_size=params['batch_size'],
                                             shuffle=True,
                                             num_workers=0,
                                             )

    # perform evaluation
    model.eval()
    valid_loss = 0
    valid_mask_loss = 0
    valid_regr_loss = 0
    with torch.no_grad():
        for img_batch, mask_batch, regr_batch in dev_loader:
            img_batch = img_batch.to(device)
            mask_batch = mask_batch.to(device)
            regr_batch = regr_batch.to(device)
            output = model(img_batch)

            if idx_epoch < params['idx_epoch_switch']:
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
    valid_loss /= len(dev_loader.dataset)
    valid_mask_loss /= len(dev_loader.dataset)
    valid_regr_loss /= len(dev_loader.dataset)
    print('Dev loss: {:.4f}'.format(valid_loss))

    # return loss as dataframe
    history = pd.DataFrame()
    history.loc[idx_epoch, 'dev_loss'] = valid_loss.cpu().numpy()
    history.loc[idx_epoch, 'mask_loss'] = valid_mask_loss.cpu().numpy()
    history.loc[idx_epoch, 'regr_loss'] = valid_regr_loss.cpu().numpy()
    return history


def train(model,
          device,
          dataset_torch,
          params,
          ):
    # Create data generators - they will produce batches
    data_loader_train = torch.utils.data.DataLoader(dataset=dataset_torch,
                                                    batch_size=params['batch_size'],
                                                    shuffle=True,
                                                    num_workers=0,
                                                    )

    # define optimizer
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=params['learning_rate'],
                                  )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(params['num_epochs'], 10) * len(data_loader_train) // 3,
        gamma=0.1,
    )

    # perform training
    df_out = pd.DataFrame()
    for idx_epoch in range(params['num_epochs']):
        torch.cuda.empty_cache()
        gc.collect()
        model.train()

        data_loader_train_tqdm =
        for batch_idx, (img_batch, mask_batch, regr_batch) in enumerate(data_loader_train_tqdm):
            img_batch = img_batch.to(device)
            mask_batch = mask_batch.to(device)
            regr_batch = regr_batch.to(device)

            optimizer.zero_grad()
            output = model(img_batch)
            if idx_epoch < params['idx_epoch_switch']:
                loss, mask_loss, regr_loss = criterion(output, mask_batch, regr_batch, 1)
            else:
                loss, mask_loss, regr_loss = criterion(output, mask_batch, regr_batch, 0.5)

            data_loader_train_tqdm.set_description(f'train_loss (l={loss:.3f})(m={mask_loss:.2f}) (r={regr_loss:.4f}')
            df_out.loc[idx_epoch + batch_idx / len(data_loader_train), 'train_loss'] = loss.data.cpu().numpy()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        print('Train Epoch: {} \tLR: {:.6f}\tLoss: {:.6f}\tMaskLoss: {:.6f}\tRegLoss: {:.6f}'.format(
            idx_epoch,
            optimizer.state_dict()['param_groups'][0]['lr'],
            loss.data,
            mask_loss.data,
            regr_loss.data,
        ))

        # evaluate training and save results        
        # evaluate(epoch, history) # commented out, because have not defined a valid dataset
        torch.save(model.state_dict(), 'model_{}.pth'.format(idx_epoch))
        df_out['train_loss'].iloc[100:].plot()
    return df_out
