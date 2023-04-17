import torch
#import albumentations as A
#from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from architecture import UNet, padding
from data_loader import LakesDataset
from utils import DEVICE, WANDB_PROJECT_NAME
import warnings
warnings.filterwarnings("ignore")

from utils import(
    load_checkpoint, 
    save_checkpoint, 
    check_accuracy, 
    save_predictions_as_imgs
)


LEARNING_RATE = 1e-4
BATCH_SIZE = 5
NUM_EPOCHS = 3
NUM_WORKERS = 2


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        # forward
        if DEVICE == "cuda":
            with torch.cuda.amp.autocast():
                predictions = padding(model(data), data)
                loss = loss_fn(predictions, targets)

        else:
            predictions = padding(model(data), data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_dataset = LakesDataset(train=True, patch_size=1024)
    val_dataset = LakesDataset(train=True, val=True, patch_size=1024)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        checkpoint = {
            "state_dict": model.state_dict(), 
            "optimizer":optimizer.state_dict()
        }

        save_checkpoint(checkpoint)

        check_accuracy(val_loader, model)

        save_predictions_as_imgs(val_loader, model)

    

if __name__ == "__main__":
    main()