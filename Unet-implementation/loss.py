import torch
import torch.nn as nn


class Dice_coeff(nn.Module):
    def __init__(
        self, act_as_loss=True, use_square=True, coeff_type="soft", threshold=0.5
    ):
        """ 
        Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
        Assumes the `channels_last` format.

        # Arguments
            act_as_loss: use it as a loss function for training
            use_square: for quantifying norm of inputs use square of the values or not
            coeff_type: use "hard" coefficient or "soft" coefficient
                "hard": convert outputs to a binary
                "soft": use outputs as it is

            threshold: Used with when performing "hard" loss. Default: 0.5
            epsilon: Used for numerical stability to avoid divide by zero errors

        # References
            V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
            https://arxiv.org/abs/1606.04797
            More details on Dice loss formulation 
            https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)

            Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
        """

        super().__init__()

        self.act_as_loss = act_as_loss
        self.use_square = use_square
        self.coeff_type = coeff_type
        self.threshold = threshold
        self.epsilon = 1e-6

    def forward(self, ground, prediction):
        """
        Arguments:
            ground: ground truth of shape (batch_size, num_classes, width, height)
            prediction: normalized prediction of shape (batch_size, num_classes, width, height)
        """

        assert ground.shape == prediction.shape

        if self.coeff_type == "hard":
            prediction = torch.sigmoid(prediction)
            prediction = (prediction > self.threshold).float()

        axes = tuple(range(2, len(prediction.shape)))
        numerator = 2.0 * (prediction * ground).sum(axis=axes)

        if self.use_square:
            denominator = torch.square(prediction) + torch.square(ground)
        else:
            denominator = prediction + ground

        denominator = denominator.sum(axis=axes)

        dice_score = ((numerator + self.epsilon) / (denominator + self.epsilon)).mean()

        if self.act_as_loss:
            return torch.tensor(1.0) - dice_score
        else:
            return dice_score


if __name__ == "__main__":
    import numpy as np

    # batch - 10, classes - 3, height, width
    shape = (128, 1, 256, 256)
    prediction = np.random.uniform(low=0, high=1, size=shape)
    ground_truth = np.random.choice([0, 1], size=shape, replace=True)

    loss_pytorch = Dice_coeff(act_as_loss=True, coeff_type="soft")
    loss = loss_pytorch(torch.from_numpy(ground_truth), torch.from_numpy(prediction))
    print(loss.item())

    