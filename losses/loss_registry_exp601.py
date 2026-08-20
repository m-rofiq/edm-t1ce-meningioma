import torch
import torch.nn.functional as F
import torchvision.models as models

def masked_l1(pred, target, mask):
    return F.l1_loss(pred[mask], target[mask])


def masked_mse(pred, target, mask):
    return F.mse_loss(pred[mask], target[mask])


def charbonnier(pred, target, mask, eps=1e-3):
    diff = pred - target
    loss = torch.sqrt(diff * diff + eps * eps)
    return loss[mask].mean()


def edge_loss(pred, target, mask):

    sobel_x = torch.tensor(
        [[-1,0,1],[-2,0,2],[-1,0,1]],
        dtype=torch.float32,
        device=pred.device
    ).view(1,1,3,3)

    sobel_y = torch.tensor(
        [[-1,-2,-1],[0,0,0],[1,2,1]],
        dtype=torch.float32,
        device=pred.device
    ).view(1,1,3,3)

    pred_x = F.conv2d(pred, sobel_x, padding=1)
    pred_y = F.conv2d(pred, sobel_y, padding=1)

    gt_x = F.conv2d(target, sobel_x, padding=1)
    gt_y = F.conv2d(target, sobel_y, padding=1)

    pred_edge = torch.sqrt(pred_x**2 + pred_y**2)
    gt_edge = torch.sqrt(gt_x**2 + gt_y**2)

    return F.l1_loss(pred_edge[mask], gt_edge[mask])

class VGGPerceptual(torch.nn.Module):

    def __init__(self):
        super().__init__()

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        self.features = torch.nn.Sequential(*list(vgg.features)[:16]).eval()

        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, pred, target):

        pred = pred.repeat(1,3,1,1)
        target = target.repeat(1,3,1,1)

        pred_f = self.features(pred)
        target_f = self.features(target)

        return F.l1_loss(pred_f, target_f)

perceptual_model = VGGPerceptual()

def perceptual_loss(pred, target, mask):

    pred = pred * mask
    target = target * mask

    return perceptual_model(pred, target)

class EDMLoss(torch.nn.Module):
    def __init__(self, lambda_enh=1.0, lambda_base=1.0):
        super().__init__()
        self.lambda_enh = lambda_enh
        self.lambda_base = lambda_base

    def forward(self, pred_tuple, target, mask, input_tensor):

        pred, base, enh = pred_tuple

        # =========================
        # T1 CENTER
        # =========================
        t1 = input_tensor[:, 1:2, :, :]

        # =========================
        # MAIN LOSS
        # =========================
        loss_main = torch.mean(torch.abs(pred - target)[mask])

        # =========================
        # 🔥 BASE CONSTRAINT (CRITICAL)
        # =========================
        loss_base = torch.mean(torch.abs(base - t1)[mask])

        # =========================
        # ENHANCEMENT GT
        # =========================
        #enh_gt = target - t1    #step 4.0
        enh_gt = torch.clamp(target - t1, min=0)  #step 4.1

        # focus tumor region only
        #enh_mask = (enh_gt > 0.1) & mask  #step 4.0
        #enh_mask = (enh_gt > 0.03) & mask  #step 4.1
        #enh_mask = (enh_gt > 0.02) & mask  #step 4.2
        enh_mask = (enh_gt > 0.025) & mask  #step 4.3

        if enh_mask.sum() > 10:
            weight = 1 + 2 * enh_mask.float()
            diff = torch.abs(enh - enh_gt)
            loss_enh = torch.sum(weight * diff * enh_mask) / (enh_mask.sum() + 1e-8)
        else:
            loss_enh = torch.mean(torch.abs(enh - enh_gt)[mask])

        return loss_main + self.lambda_base * loss_base + self.lambda_enh * loss_enh

LOSS_REGISTRY = {

    "MaskedL1": masked_l1,
    "MaskedMSE": masked_mse,
    "Charbonnier": charbonnier,
    "EdgeLoss": edge_loss,
    "Perceptual": perceptual_loss,
    #"EDMLoss": EDMLoss(lambda_enh=1.0, lambda_base=1.0) #step 4.0
    #"EDMLoss": EDMLoss(lambda_enh=1.3, lambda_base=1.0)  #step 4.1
    #"EDMLoss": EDMLoss(lambda_enh=2.0, lambda_base=1.0)  #step 4.2
    "EDMLoss": EDMLoss(lambda_enh=1.6, lambda_base=1.0)  #step 4.3
    
}