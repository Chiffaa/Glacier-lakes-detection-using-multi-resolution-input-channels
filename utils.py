import torch
import torchvision
from torchmetrics import JaccardIndex, Accuracy, Dice, Recall, Precision
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

    acc_score = 0
    precision_score = 0
    recall_score = 0
    dice_score = 0
    iou_score = 0

    
    accuracy = Accuracy(task='binary', num_classes=2).to(DEVICE)
    recall = Recall(task = 'binary', num_classes=2).to(DEVICE)
    precision = Precision(task = 'binary', num_classes=2).to(DEVICE)
    # dice = Dice(num_classes=2)
    iou = JaccardIndex(task='binary', num_classes=2).to(DEVICE)

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = padding(torch.sigmoid(model(x)), x)

            acc_score += accuracy(preds, y)
            precision_score += precision(preds, y)
            recall_score += recall(preds, y)
            # dice_score += dice(preds, y)
            iou_score += iou(preds, y)

            

    #print(f"{dataset}_acc: {acc_score/len(loader)},{dataset}_precision': {precision_score/len(loader)},{dataset}_recall': {recall_score/len(loader)},{dataset}_IoU_score': {iou_score/len(loader)}")

    wandb.log({f'{dataset}_acc':acc_score/len(loader), 
               f'{dataset}_precision': precision_score/len(loader), 
               f'{dataset}_recall': recall_score/len(loader),  
               f'{dataset}_IoU_score':iou_score/len(loader)})

    model.train()


def save_predictions_as_imgs(loader, model, folder="saved_images/", device=DEVICE):
    model.eval()
    for idx, (x,y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = (torch.sigmoid(model(x)) > 0.5).float()
            preds = padding(preds, x)
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")

        




