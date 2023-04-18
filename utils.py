import torch
import torchvision
from data_loader import LakesDataset
from architecture import padding
import wandb

WANDB_PROJECT_NAME = "glacier-lakes-mapping"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def check_accuracy(loader, model, device=DEVICE):
    num_correct = 0
    num_pixels = 0

    dice_score = 0

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = (torch.sigmoid(model(x)) > 0.5).float()
            preds = padding(preds, x)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2*(preds*y).sum()) / ((preds + y).sum() + 1e-8)

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels * 100:.2f}%"
    )
    print(f"Dice score: {dice_score/len(loader)}")

    wandb.log({'val_acc':num_correct/num_pixels * 100, 'dice_score':dice_score/len(loader)})

    model.train()


def save_predictions_as_imgs(loader, model, folder="saved_images/", device=DEVICE):
    model.eval()
    for idx, (x,y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = (torch.sigmoid(model(x)) > 0.5).float()
            preds = padding(preds, x)
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")

        




