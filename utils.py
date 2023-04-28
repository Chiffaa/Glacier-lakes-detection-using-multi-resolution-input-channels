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

def check_accuracy(loaders, model, val=True, device=DEVICE):

    train_acc_score = 0
    train_precision_score = 0
    train_recall_score = 0
    train_dice_score = 0
    train_iou_score = 0

    val_acc_score = 0
    val_precision_score = 0
    val_recall_score = 0
    val_dice_score = 0
    val_iou_score = 0

    train_loader = loaders['train_loader']
    val_loader = loaders['val_loader']

    
    accuracy = Accuracy(task='binary', num_classes=2).to(DEVICE)
    recall = Recall(task = 'binary', num_classes=2).to(DEVICE)
    precision = Precision(task = 'binary', num_classes=2).to(DEVICE)
    # dice = Dice(num_classes=2)
    iou = JaccardIndex(task='binary', num_classes=2).to(DEVICE)

    model.eval()

    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            preds = padding(torch.sigmoid(model(x)), x)

            train_acc_score += accuracy(preds, y)
            train_precision_score += precision(preds, y)
            train_recall_score += recall(preds, y)
            # train_dice_score += dice(preds, y)
            train_iou_score += iou(preds, y)

        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            preds = padding(torch.sigmoid(model(x)), x)

            val_acc_score += accuracy(preds, y)
            val_precision_score += precision(preds, y)
            val_recall_score += recall(preds, y)
            # val_dice_score += dice(preds, y)
            val_iou_score += iou(preds, y)

            

    #print(f"{dataset}_acc: {acc_score/len(loader)},{dataset}_precision': {precision_score/len(loader)},{dataset}_recall': {recall_score/len(loader)},{dataset}_IoU_score': {iou_score/len(loader)}")

    wandb.log({f'train_acc':train_acc_score/len(train_loader), 
               f'train_precision': train_precision_score/len(train_loader), 
               f'train_recall': train_recall_score/len(train_loader),  
               f'train_IoU_score':train_iou_score/len(train_loader), 
               f'val_acc':val_acc_score/len(val_loader), 
               f'val_precision': val_precision_score/len(val_loader), 
               f'val_recall': val_recall_score/len(val_loader),  
               f'val_IoU_score':val_iou_score/len(val_loader),})

    model.train()


def save_predictions_as_imgs(loader, model, folder="saved_images/", device=DEVICE):
    model.eval()
    for idx, (x,y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = (torch.sigmoid(model(x)) > 0.5).float()
            preds = padding(preds, x)
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")

        




