import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class VGGPerceptualLoss(nn.Module):
    """
    Computes perceptual loss using a pretrained VGG-16 network.
    Uses features from intermediate layers (e.g., relu1_2, relu2_2, relu3_3, relu4_3).
    """
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        self.enabled = True
        try:
            # Load VGG16 weights
            vgg = vgg16(weights=VGG16_Weights.DEFAULT)
            # We only need the features part, and we freeze it
            vgg_features = vgg.features.eval()
            for param in vgg_features.parameters():
                param.requires_grad = False
                
            self.slice1 = nn.Sequential(*vgg_features[:4])   # relu1_2
            self.slice2 = nn.Sequential(*vgg_features[4:9])  # relu2_2
            self.slice3 = nn.Sequential(*vgg_features[9:16]) # relu3_3
            self.slice4 = nn.Sequential(*vgg_features[16:23]) # relu4_3
            
            # Mean and std normalization for VGG input (expecting range [0, 1])
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        except Exception as e:
            print(f"Warning: Could not load pretrained VGG-16 for perceptual loss ({e}). Perceptual loss is disabled.")
            self.enabled = False

    def _normalize(self, x):
        # Convert from [-1, 1] to [0, 1] before normalizing
        x = (x + 1.0) / 2.0
        return (x - self.mean) / self.std

    def forward(self, x, y):
        if not self.enabled:
            return torch.tensor(0.0, device=x.device, requires_grad=True)
            
        x = self._normalize(x)
        y = self._normalize(y)
        
        # If input has 1 channel (e.g., grayscale/IR) replicate to 3 channels
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        if y.size(1) == 1:
            y = y.repeat(1, 3, 1, 1)
            
        h_x1 = self.slice1(x)
        h_y1 = self.slice1(y)
        h_x2 = self.slice2(h_x1)
        h_y2 = self.slice2(h_y1)
        h_x3 = self.slice3(h_x2)
        h_y3 = self.slice3(h_y2)
        h_x4 = self.slice4(h_x3)
        h_y4 = self.slice4(h_y3)
        
        loss = (
            nn.functional.l1_loss(h_x1, h_y1) +
            nn.functional.l1_loss(h_x2, h_y2) +
            nn.functional.l1_loss(h_x3, h_y3) +
            nn.functional.l1_loss(h_x4, h_y4)
        )
        return loss

class GANLoss(nn.Module):
    """
    Adversarial loss: LSGAN (MSE) or Vanilla GAN (BCEWithLogits)
    """
    def __init__(self, use_lsgan=True):
        super(GANLoss, self).__init__()
        self.loss = nn.MSELoss() if use_lsgan else nn.BCEWithLogitsLoss()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            return torch.ones_like(prediction)
        else:
            return torch.zeros_like(prediction)

    def forward(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)
