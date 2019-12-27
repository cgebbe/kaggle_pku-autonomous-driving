import torch
import model
import torch.optim as optim
from torch.optim import lr_scheduler


def train(model):
    n_epochs = 12  # 6
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    # optimizer =  RAdam(model.parameters(), lr = 0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=max(n_epochs, 10) * len(train_loader) // 3,
        gamma=0.1,
    )



