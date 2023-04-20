import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from architecture import UNet, padding
from data_loader import LakesDataset
from utils import DEVICE, WANDB_PROJECT_NAME
import wandb
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
BATCH_SIZE=32

def train_fn(data, targets, model, optimizer, loss_fn):
    
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
    loss.backward()

    #optimizer step
    optimizer.step()

    
    return loss.item()

def train_log(loss, batch, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "batch":batch, "loss": loss})


def train(model, loaders, loss_fn, optimizer, epochs=NUM_EPOCHS):
    with wandb.init(project='test_launch'):
        wandb.config = {"learning_rate": LEARNING_RATE, "epochs": epochs, "batch_size": BATCH_SIZE}
        wandb.watch(model) 
        
        loop = tqdm(loaders['train_loader'])

        for epoch in range(epochs):

            for batch_idx, (data, targets) in enumerate(loop):

                loss = train_fn(data, targets, model, optimizer, loss_fn)


                # update tqdm loop
                loop.set_postfix(loss=loss)

                train_log(loss, batch_idx, epoch)

            check_accuracy(loaders['val_loader'], model)       
        wandb.save("model.onnx")




def main():
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_dataset = LakesDataset(train=True, patch_size=1024)
    val_dataset = LakesDataset(train=False, val=True, patch_size=1024)

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