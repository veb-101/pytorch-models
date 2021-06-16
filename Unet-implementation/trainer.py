import os
import gc
import shutil
import numpy as np
from tqdm.notebook import tqdm

import torch
from torch.cuda import amp
from torch.optim.adam import Adam
from torchvision.utils import save_image

from model import UNet
from loss import Dice_coeff
from utils import denormalize, get_device, check_accuracy, save_predictions_as_imgs

# import wandb


class Trainer:
    def __init__(self, config, data_loader_train, data_loader_val):
        self.device = get_device()
        self.num_classes = config["num_classes"]
        self.num_epoch = config["num_epoch"]
        self.start_epoch = config["start_epoch"]
        self.image_size = config["image_size"]
        self.save_dir = config["save_dir"]

        self.batch_size = config["batch_size"]
        self.data_loader_train = data_loader_train
        self.data_loader_val = data_loader_val
        self.lr = config["lr"]
        self.decay_iter = config["decay_iter"]

        self.metrics = {
            "dice_loss": [],
            "val_dice_score": [],
            "val_accuracy": [],
        }

        self.build_model(config)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, self.decay_iter
        )

    def train(self):
        os.makedirs("/content/drive/MyDrive/Project-UNet-Carvana", exist_ok=True)

        total_step = len(self.data_loader_train)

        best_val_dice = 0.0
        best_val_acc = 0.0

        for epoch in range(self.start_epoch, self.start_epoch + self.num_epoch):
            self.model.train()

            epoch_loss = []

            SAVE = False

            training_loader_iter = iter(self.data_loader_train)
            length_train = len(training_loader_iter)

            if not os.path.exists(os.path.join(self.save_dir, str(epoch))):
                os.makedirs(os.path.join(self.save_dir, str(epoch)))

            for step in tqdm(
                range(length_train),
                desc=f"Epoch: {epoch}/{self.start_epoch + self.num_epoch-1}",
            ):
                image, ground_mask = next(training_loader_iter)

                self.optimizer.zero_grad()

                with amp.autocast():
                    predicted_mask = self.generator(image)
                    dice_loss = self.loss_func(ground_mask, predicted_mask)

                self.scaler.scale(dice_loss).backward()
                self.scaler.step(self.optimizer)

                scale = self.scaler.get_scale()
                self.scaler.update()
                skip_lr_sched = scale != self.scaler.get_scale()
                # self.optimizer_generator.step()

                self.metrics["dice_loss"].append(np.round(dice_loss.detach().item(), 5))

                epoch_loss.append(self.metrics["dice_loss"][-1])

                torch.cuda.empty_cache()
                gc.collect()

                if step == 0 or step == total_step // 2 or step == (total_step - 1):

                    print(
                        f"[Epoch {epoch}/{self.start_epoch+self.num_epoch-1}] [Batch {step+1}/{total_step}]"
                        f"[dice loss {self.metrics['dice_loss'][-1]}]"
                        f""
                    )

                    result = torch.cat(
                        (
                            image.detach().cpu(),
                            ground_mask.detach().cpu(),
                            predicted_mask.detach().cpu(),
                        ),
                        2,
                    )

                    if self.normalized:
                        result = denormalize(result)

                    save_image(
                        result,
                        os.path.join(self.save_dir, str(epoch), f"UNet_{step+1}.png"),
                        nrow=8,
                        normalize=False,
                    )
                    # wandb.log(
                    #     {
                    #         f"training_image_{step+1}": wandb.Image(
                    #             os.path.join(
                    #                 self.save_dir, str(epoch), f"UNet_{step+1}.png"
                    #             )
                    #         )
                    #     }
                    # )

                torch.cuda.empty_cache()
                gc.collect()

            # epoch metrics
            print(
                f"Epoch: {epoch} -> "
                f"Dice loss: {np.round(np.array(epoch_loss).mean(), 4)} "
                f""
            )
            # wandb.log(
            #     {"epoch": epoch, "dice_loss": np.round(np.array(epoch_loss).mean(), 4),}
            # )

            if not skip_lr_sched:
                self.lr_scheduler.step()

            # validation set SSIM and PSNR
            val_dice_score, val_acc = check_accuracy(
                self.data_loader_val, self.model, self.device
            )

            self.metrics["val_dice_score"].append(val_dice_score)
            self.metrics["val_accuracy"].append(val_acc)

            # log validation psnr, ssim
            # wandb.log(
            #     {
            #         "epoch": epoch,
            #         "valid_dice_score": val_dice_score,
            #         "valid_acc": val_acc,
            #     }
            # )

            # visualization
            if not epoch % 5:
                save_predictions_as_imgs(
                    self.data_loader_val,
                    self.model,
                    folder=self.save_dir,
                    device=self.deivce,
                )
                # log validation image 0

                # image_1
                # image_2
                # image_3
                # result = torch.cat
                # wandb.log(
                #     {
                #         "validation_images": wandb.Image(
                #             os.path.join(self.save_dir, f"Validation_{epoch}_0.png")
                #         ),
                #     }
                # )

            if val_dice_score > best_val_dice:
                best_val_dice = val_dice_score
                SAVE = True

            print(
                f"Validation Set: Dice Score: {val_dice_score}, Accuracy: {val_acc * 100:.2f}"
            )

            del val_dice_score, val_acc

            torch.cuda.empty_cache()
            gc.collect()

            models_dict = {
                "next_epoch": epoch + 1,
                f"model": self.model.state_dict(),
                f"optimizer": self.optimizer.state_dict(),
                f"grad_scaler": self.scaler.state_dict(),
                f"metrics": self.metrics,
            }
            save_name = f"checkpoint_{epoch}.tar"

            if SAVE:
                if save_name.startswith("checkpoint"):
                    _ = [
                        os.remove(os.path.join(r"/content/", file))
                        for file in os.listdir(r"/content/")
                        if file.startswith("checkpoint")
                    ]

                torch.save(models_dict, save_name)
                shutil.copyfile(
                    save_name,
                    os.path.join(
                        r"/content/drive/MyDrive/Project-UNet-Carvana", save_name
                    ),
                )

            torch.cuda.empty_cache()
            gc.collect()

        # wandb.run.summary["valid_dice_score"] = best_val_dice
        # wandb.run.summary["valid_acc"] = best_val_acc

        return self.metrics

    def build_model(self, config):

        self.model = UNet(in_channels=3, out_channels=self.num_classes)
        self.optimizer = Adam(
            self.generator.parameters(),
            lr=self.lr,
            betas=(config["b1"], config["b2"]),
            weight_decay=config["weight_decay"],
        )
        self.loss_func = Dice_coeff(
            act_as_loss=True, use_square=True, coeff_type="soft"
        )

        self.scaler = torch.cuda.amp.GradScaler()

        self.load_model()

    def load_model(self,):
        drive_path = r"/content/drive/MyDrive/Project-UNet-Carvana"
        print(f"[*] Finding checkpoint {self.start_epoch-1} in {drive_path}")

        checkpoint_file = f"checkpoint_{self.start_epoch-1}.tar"
        checkpoint_path = os.path.join(drive_path, checkpoint_file)
        if not os.path.exists(checkpoint_path):
            print(f"[!] No checkpoint for epoch {self.start_epoch -1}")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint[f"model"])
        print("Model weights loaded.")

        self.optimizer_generator.load_state_dict(checkpoint[f"optimizer"])
        print("Optimizer state loaded")

        self.scaler_gen.load_state_dict(checkpoint[f"grad_scaler"])
        print("Grad Scaler - loaded")

        self.start_epoch = checkpoint["next_epoch"]

        temp = []

        if self.decay_iter:
            self.decay_iter = np.array(self.decay_iter) - self.start_epoch

            for i in self.decay_iter:
                if i > 0:
                    temp.append(i)

        if not temp:
            temp.append(200)

        self.decay_iter = temp
        print("Decay_iter:", self.decay_iter)

        print(f"Checkpoint: {self.start_epoch-1} loaded")
