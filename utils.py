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

def check_accuracy(loader, model, val=True, device=DEVICE):

    if val: 
        dataset = 'val'
    else:
        dataset = 'train'

    num_correct = 0
    num_pixels = 0
    precision = 0
    recall = 0
    dice_score = 0
    iou = 0

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = (torch.sigmoid(model(x)) > 0.5).float()
            preds = padding(preds, x)
            num_correct += (preds == y).sum() # True Positives
            num_pixels += torch.numel(preds)

            # Metrics Precision, Recall, Dice score, IoU
            precision += num_correct / preds.sum() # preds.sum() = TP + FP
            recall += num_correct / y.sum() # y.sum() = TP + FN

            # 2*TP / ((TP + FP) + (TP + FN))
            dice_score += (2*(preds*y).sum()) / ((preds + y).sum() + 1e-8)

            # Intersection over Union
            intersection = (preds * y).sum()
            union = preds.sum() + y.sum() - intersection
            iou += intersection / union

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels * 100:.2f}%"
    )
    print(f"Dice score: {dice_score/len(loader)}, IoU score: {iou/len(loader)}")

    wandb.log({f'{dataset}_acc':num_correct/num_pixels, 
               f'{dataset}_precision': precision/len(loader), 
               f'{dataset}_recall': recall/len(loader), 
               f'{dataset}_dice_score':dice_score/len(loader), 
               f'{dataset}_IoU_score':iou/len(loader)})

    model.train()


def save_predictions_as_imgs(loader, model, folder="saved_images/", device=DEVICE):
    model.eval()
    for idx, (x,y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = (torch.sigmoid(model(x)) > 0.5).float()
            preds = padding(preds, x)
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")

        




