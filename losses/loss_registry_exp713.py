import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ==================================================
# BASIC
# ==================================================
def masked_l1(pred, target, mask):
    return F.l1_loss(pred[mask], target[mask])


def masked_mse(pred, target, mask):
    return F.mse_loss(pred[mask], target[mask])


# ==================================================
# EDGE
# ==================================================
def edge_loss(pred, target, mask):
    sobel_x = torch.tensor(
        [[-1,0,1],[-2,0,2],[-1,0,1]],
        dtype=pred.dtype, device=pred.device
    ).view(1,1,3,3)

    sobel_y = torch.tensor(
        [[-1,-2,-1],[0,0,0],[1,2,1]],
        dtype=pred.dtype, device=pred.device
    ).view(1,1,3,3)

    px = F.conv2d(pred, sobel_x, padding=1)
    py = F.conv2d(pred, sobel_y, padding=1)
    gx = F.conv2d(target, sobel_x, padding=1)
    gy = F.conv2d(target, sobel_y, padding=1)

    pe = torch.sqrt(px.pow(2)+py.pow(2)+1e-8)
    ge = torch.sqrt(gx.pow(2)+gy.pow(2)+1e-8)

    return F.l1_loss(pe[mask], ge[mask])


# ==================================================
# LAPLACIAN
# ==================================================
def laplacian_loss(pred, target, mask):
    kernel = torch.tensor(
        [[0,-1,0],
         [-1,4,-1],
         [0,-1,0]],
        dtype=pred.dtype, device=pred.device
    ).view(1,1,3,3)

    p = F.conv2d(pred, kernel, padding=1)
    t = F.conv2d(target, kernel, padding=1)

    return F.l1_loss(p[mask], t[mask])


# ==================================================
# VGG
# ==================================================
class VGGPerceptual(nn.Module):
    def __init__(self):
        super().__init__()

        vgg = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1
        ).features.eval()

        for p in vgg.parameters():
            p.requires_grad = False

        self.s1 = nn.Sequential(*list(vgg)[:4])
        self.s2 = nn.Sequential(*list(vgg)[4:9])
        self.s3 = nn.Sequential(*list(vgg)[9:16])

    def forward(self, pred, target):
        pred = pred.repeat(1,3,1,1)
        target = target.repeat(1,3,1,1)

        p1 = self.s1(pred); t1 = self.s1(target)
        p2 = self.s2(p1);   t2 = self.s2(t1)
        p3 = self.s3(p2);   t3 = self.s3(t2)

        return (
            F.l1_loss(p1,t1)+
            F.l1_loss(p2,t2)+
            F.l1_loss(p3,t3)
        ) / 3.0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
perceptual_model = None


# ==================================================
# EXP-708
# ==================================================
class EDMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_tuple, target, mask, input_tensor):
        pred, base, enh = pred_tuple
        mask_f = mask.float()

        t1 = input_tensor[:,1:2,:,:]
        diff = target - t1

        healthy = (diff <= 0.025) & mask
        tumor   = (diff > 0.025) & mask

        loss_global = torch.mean(torch.abs(pred-target)[mask])

        loss_tumor = (
            torch.mean(torch.abs(pred-target)[tumor])
            if tumor.any() else pred.new_tensor(0.0)
        )

        loss_base = (
            torch.mean(torch.abs(base-target)[healthy])
            if healthy.any() else pred.new_tensor(0.0)
        )

        enh_gt = torch.clamp(target - base.detach(), min=0.0)

        loss_enh = (
            torch.mean(torch.abs(enh-enh_gt)[tumor])
            if tumor.any()
            else torch.mean(torch.abs(enh-enh_gt)[mask])
        )

        loss_edge = edge_loss(pred, target, mask)
        loss_lap  = laplacian_loss(pred, target, mask)

        global perceptual_model

        if perceptual_model is None:
            perceptual_model = VGGPerceptual().to(pred.device).eval()

        if pred.device.type == "cuda":
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    loss_vgg = perceptual_model(
                        (pred * mask_f).float(),
                        (target * mask_f).float()
                    )
        else:
            with torch.no_grad():
                loss_vgg = perceptual_model(
                    (pred * mask_f).float(),
                    (target * mask_f).float()
                )

        total = (
            0.35 * loss_global +
            1.50 * loss_tumor +
            1.00 * loss_base +
            1.20 * loss_enh +
            0.08 * loss_edge +
            0.10 * loss_lap +
            0.08 * loss_vgg
        )

        return total


LOSS_REGISTRY = {
    "EDMLoss": EDMLoss()
}