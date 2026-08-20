import torch
import torch.nn.functional as F
import torchvision.models as models
from ddresunet_loss import ddresunet_loss

# def masked_l1(pred, target, mask):
#     return F.l1_loss(pred[mask], target[mask])


# def masked_mse(pred, target, mask):
#     return F.mse_loss(pred[mask], target[mask])


# def charbonnier(pred, target, mask, eps=1e-3):
#     diff = pred - target
#     loss = torch.sqrt(diff * diff + eps * eps)
#     return loss[mask].mean()


# def edge_loss(pred, target, mask):

#     sobel_x = torch.tensor(
#         [[-1,0,1],[-2,0,2],[-1,0,1]],
#         dtype=torch.float32,
#         device=pred.device
#     ).view(1,1,3,3)

#     sobel_y = torch.tensor(
#         [[-1,-2,-1],[0,0,0],[1,2,1]],
#         dtype=torch.float32,
#         device=pred.device
#     ).view(1,1,3,3)

#     pred_x = F.conv2d(pred, sobel_x, padding=1)
#     pred_y = F.conv2d(pred, sobel_y, padding=1)

#     gt_x = F.conv2d(target, sobel_x, padding=1)
#     gt_y = F.conv2d(target, sobel_y, padding=1)

#     pred_edge = torch.sqrt(pred_x**2 + pred_y**2)
#     gt_edge = torch.sqrt(gt_x**2 + gt_y**2)

#     return F.l1_loss(pred_edge[mask], gt_edge[mask])

# class VGGPerceptual(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.eval()
#         for p in vgg.parameters():
#             p.requires_grad = False
            
#         # Potong menjadi 3 blok untuk menangkap detail tepi dan struktur
#         self.slice1 = torch.nn.Sequential(*list(vgg)[:4])   # relu1_2 (High-freq/Tekstur)
#         self.slice2 = torch.nn.Sequential(*list(vgg)[4:9])  # relu2_2 (Mid-freq)
#         self.slice3 = torch.nn.Sequential(*list(vgg)[9:16]) # relu3_3 (Low-freq/Struktur)

#     def forward(self, pred, target):
#         pred_rgb = pred.repeat(1,3,1,1)
#         target_rgb = target.repeat(1,3,1,1)

#         # Ekstraksi Prediksi
#         h1_pred = self.slice1(pred_rgb)
#         h2_pred = self.slice2(h1_pred)
#         h3_pred = self.slice3(h2_pred)

#         # Ekstraksi Target
#         h1_gt = self.slice1(target_rgb)
#         h2_gt = self.slice2(h1_gt)
#         h3_gt = self.slice3(h2_gt)

#         # Gabungkan loss (bisa dirata-rata)
#         loss = F.l1_loss(h1_pred, h1_gt) + F.l1_loss(h2_pred, h2_gt) + F.l1_loss(h3_pred, h3_gt)
#         return loss / 3.0

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# perceptual_model = VGGPerceptual().to(device)

# def perceptual_loss(pred, target, mask):

#     pred = pred * mask
#     target = target * mask

#     return perceptual_model(pred, target)

# class EDMLoss(torch.nn.Module):
#     def __init__(self, lambda_enh=1.6, lambda_base=1.0, lambda_vgg=0.05):
#         super().__init__()
#         self.lambda_enh = lambda_enh
#         self.lambda_base = lambda_base
#         self.lambda_vgg = lambda_vgg

#     def forward(self, pred_tuple, target, mask, input_tensor):
#         pred, base, enh = pred_tuple

#         # =========================
#         # MAIN LOSS
#         # =========================
#         loss_main = torch.mean(torch.abs(pred - target)[mask])

#         # =========================
#         # BASE / ENH SPLIT
#         # =========================
#         t1 = input_tensor[:, 1:2, :, :]

#         diff = target - t1
#         healthy_mask = (diff <= 0.025) & mask
#         tumor_mask = (diff > 0.025) & mask

#         # =========================
#         # BASE LOSS
#         # =========================
#         if healthy_mask.sum() > 0:
#             loss_base = torch.mean(torch.abs(base - target)[healthy_mask])
#         else:
#             loss_base = torch.tensor(0.0, device=pred.device)

#         # =========================
#         # ENH LOSS
#         # =========================
#         #enh_gt = target - base.detach()
#         enh_gt = torch.clamp(target - t1, min=0)
#         #enh_gt = torch.clamp(enh_gt, min=0)

#         if tumor_mask.sum() > 10:
#             loss_enh = torch.mean(torch.abs(enh - enh_gt)[tumor_mask])
#             #enh_mask = tumor_mask
#         else:
#             loss_enh = torch.mean(torch.abs(enh - enh_gt)[mask])
#             #enh_mask = mask
        
#         # =========================
#         # VGG PERCEPTUAL LOSS (ON FULL PRED)
#         # =========================
#         pred_masked = pred * mask.float()
#         target_masked = target * mask.float()
        
#         #with torch.amp.autocast(device_type='cuda', enabled=False):
#         with torch.amp.autocast(device_type=pred.device.type, enabled=False):
#             loss_vgg = perceptual_model(pred_masked.float(), target_masked.float())

#         # =========================
#         # TOTAL LOSS
#         # =========================
#         # Kurangi sedikit dominasi loss_main (misal beri bobot 0.5) jika masih blur
#         return (
#             0.5 * loss_main
#             + self.lambda_base * loss_base
#             + self.lambda_enh * loss_enh 
#             + self.lambda_vgg * loss_vgg
#         )

LOSS_REGISTRY = {

    # "MaskedL1": masked_l1,
    # "MaskedMSE": masked_mse,
    # "Charbonnier": charbonnier,
    # "EdgeLoss": edge_loss,
    # "Perceptual": perceptual_loss,
    #"EDMLoss": EDMLoss(lambda_enh=1.0, lambda_base=1.0) #step 4.0
    #"EDMLoss": EDMLoss(lambda_enh=1.3, lambda_base=1.0)  #step 4.1
    #"EDMLoss": EDMLoss(lambda_enh=2.0, lambda_base=1.0)  #step 4.2
    #"EDMLoss": EDMLoss(lambda_enh=1.6, lambda_base=1.0)  #step 4.3
    #"EDMLoss": EDMLoss(lambda_enh=1.6, lambda_base=1.0, lambda_vgg=0.05) #tune 1
    #"EDMLoss": EDMLoss(lambda_enh=1.6, lambda_base=1.0, lambda_vgg=0.1) #tune 2
    #"EDMLoss": EDMLoss(lambda_enh=1.0, lambda_base=1.0, lambda_vgg=0.15) #tune 3
    #"EDMLoss": EDMLoss(lambda_enh=1.0, lambda_base=1.0, lambda_vgg=0.1) #tune 4
    "DDResUNetLoss": ddresunet_loss
}