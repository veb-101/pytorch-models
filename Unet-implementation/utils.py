import os
import torch
import numpy as np
from loss import Dice_coeff
from torchvision.utils import save_image


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
#     print("=> Saving checkpoint")
#     torch.save(state, filename)


# def load_checkpoint(checkpoint, model):
#     print("=> Loading checkpoint")
#     model.load_state_dict(checkpoint["state_dict"])


def denormalize(tensors):
    """Normalization parameters for pre-trained PyTorch models
     Denormalizes image tensors using mean and std """
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    tensors = tensors.clone()
    # print(tensors[:, 0, :, :].size())
    # print(tensors[:, 1, :, :].size())
    # print(tensors[:, 2, :, :].size())

    for c in range(3):
        tensors[:, c, :, :].mul_(std[c]).add_(mean[c])

    return torch.from_numpy(np.clip(tensors.cpu().numpy(), 0, 255))


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    check_dice_acc = Dice_coeff(
        act_as_loss=False, use_square=True, coeff_type="hard", threshold=0.5
    )

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            preds = model(x)

            dice_score += check_dice_acc(y, preds)

            preds = torch.sigmoid(preds)
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

    acc = num_correct / num_pixels
    dice_score = dice_score / len(loader)
    model.train()
    return dice_score, acc


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        save_image(y.unsqueeze(1), os.path.join(f"folder", f"{idx}.png"))
        save_image(preds, os.path.join(f"{folder}, pred_{idx}.png"))

    model.train()
