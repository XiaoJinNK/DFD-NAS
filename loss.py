from torch import nn
import torch
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable


# --------------------------------------------Dice Loss-------------------------------------------
class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        N = targets.size()[0]
        smooth = 1
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)
        intersection = input_flat * targets_flat
        dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        loss = 1 - dice_eff.sum() / N
        return loss


# --------------------------------------------Focal Loss-------------------------------------------
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, epsilon=1.e-9):

        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, input, target):
        """
        Args:
            logits: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        multi_hot_key = target
        logits = torch.sigmoid(input)
        zero_hot_key = 1 - multi_hot_key
        loss = -self.alpha * multi_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
        loss += -(1 - self.alpha) * zero_hot_key * torch.pow(logits, self.gamma) * (1 - logits + self.epsilon).log()
        return loss.mean()


# -----------------------------------Weighted BCE Loss-------------------------------------------
class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, reduction='mean'):
        """
        Implementation: [2021 IS] Feature pyramid network for diffusion-based image inpainting detection
        https://codeleading.com/article/67684830894/

        loss = Nn/(Np+Nn) * pos_loss + Np/(Np+Nn) * neg_loss
        """
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        assert (input.size() == target.size())

        epsilon = 1e-10
        pos_num = torch.sum(target).to(torch.float32) + epsilon
        neg_num = torch.sum(1 - target).to(torch.float32) + epsilon

        alpha = neg_num / pos_num
        beta = pos_num / (pos_num + neg_num)

        bce = nn.BCEWithLogitsLoss(pos_weight=alpha, reduction=self.reduction)
        return beta * bce(input, target)


# --------------------------------------------IOU Loss-------------------------------------------
class IOULoss(nn.Module):

    def __init__(self, reduction='mean'):
        super(IOULoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        b = pred.shape[0]
        IoU = 0.0
        for i in range(0, b):
            Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
            Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
            IoU1 = Iand1 / Ior1

            # IoU loss is (1-IoU1)
            IoU = IoU + (1 - IoU1)

        return IoU / b


# --------------------------------------------SSIM Loss-------------------------------------------
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return 1 - ssim_map.mean()


class SSIM(torch.nn.Module):

    def __init__(self, window_size=11):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel)


class HybridLossWithLogits(nn.Module):
    """
    Implementation:
        Hybrid Loss - BCE+IOU+SSIM
            [2021 arxiv] Boundary-aware segmentation network for mobile and web applications
            [2020 TII] AR-Net: adaptive attention and residual refinement network for copy-move forgery detection
        Adaptive WBCE+BCE:
            [2021 IS] Feature pyramid network for diffusion-based image inpainting detection

    loss = adaptive(WBCE + BCE) + SSIM + IOU
    """

    def __init__(self):
        """
        alpha = epoch / T
        alpha: (1-alpha) is the weight of normal BCELoss, alpha is the weight of WBCELoss
        """
        super(HybridLossWithLogits, self).__init__()

        self.bce = nn.BCEWithLogitsLoss(pos_weight=None)
        self.ssim = SSIM()
        self.iou = IOULoss()

    def forward(self, logits, target):
        return self.bce(logits, target) + \
               self.ssim(torch.sigmoid(logits), target) + self.iou(torch.sigmoid(logits), target)



if __name__ == '__main__':
    logits = torch.randn(size=(1, 1, 256, 256))
    mask = torch.randint(low=0, high=2, size=(1, 1, 256, 256)).to(torch.float32)
    print(torch.unique(mask))
    print(nn.BCEWithLogitsLoss()(logits, mask))
    print(SSIM()(torch.sigmoid(logits), mask))
    print(IOULoss()(torch.sigmoid(logits), mask))
    print(HybridLossWithLogits()(logits, mask))

