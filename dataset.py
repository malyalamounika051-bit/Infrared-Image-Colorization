import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

def generate_synthetic_data(save_dir, num_images=150, image_size=(256, 256)):
    """
    Generates synthetic paired RGB and IR satellite-like images for training.
    """
    os.makedirs(os.path.join(save_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "ir"), exist_ok=True)

    print(f"Generating {num_images} synthetic image pairs in {save_dir}...")
    
    h, w = image_size
    
    for idx in range(num_images):
        # 1. Create RGB image
        # Background: vegetation (green with some noise/texture)
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        # Random green shades
        green_base = np.random.randint(60, 120)
        rgb[:, :, 1] = green_base # Green
        rgb[:, :, 0] = np.random.randint(20, green_base - 20, size=(h, w)) # Red
        rgb[:, :, 2] = np.random.randint(10, 50, size=(h, w)) # Blue
        
        # Simulated IR image
        ir = np.zeros((h, w), dtype=np.uint8)
        # Vegetation is highly reflective in near-IR (bright)
        ir_veg = np.random.randint(160, 210, size=(h, w), dtype=np.uint8)
        ir[:, :] = ir_veg

        # Add a water body (river or lake)
        # Water is blue in RGB, very dark/cool in IR
        if np.random.rand() > 0.3:
            water_type = np.random.choice(["lake", "river"])
            if water_type == "lake":
                cx, cy = np.random.randint(40, w - 40), np.random.randint(40, h - 40)
                rx, ry = np.random.randint(20, 60), np.random.randint(20, 60)
                # Draw lake in RGB
                cv2.ellipse(rgb, (cx, cy), (rx, ry), np.random.randint(0, 360), 0, 360, (10, 40, np.random.randint(150, 220)), -1)
                # Draw lake in IR (water is very cool/dark)
                cv2.ellipse(ir, (cx, cy), (rx, ry), np.random.randint(0, 360), 0, 360, np.random.randint(20, 50), -1)
            else:
                # River
                pts = np.array([
                    [0, np.random.randint(0, h)],
                    [w // 3, np.random.randint(0, h)],
                    [2 * w // 3, np.random.randint(0, h)],
                    [w, np.random.randint(0, h)]
                ], dtype=np.int32)
                # Draw river
                thickness = np.random.randint(8, 25)
                cv2.polylines(rgb, [pts], False, (15, 55, np.random.randint(160, 230)), thickness)
                cv2.polylines(ir, [pts], False, np.random.randint(25, 60), thickness)
                # Smooth/fill the river slightly
                rgb = cv2.GaussianBlur(rgb, (3, 3), 0)
                ir = cv2.GaussianBlur(ir, (3, 3), 0)

        # Add roads (gray/black in RGB, warm in IR)
        num_roads = np.random.randint(1, 4)
        for _ in range(num_roads):
            pt1 = (np.random.choice([0, w-1]), np.random.randint(0, h-1))
            pt2 = (np.random.randint(0, w-1), np.random.choice([0, h-1]))
            road_width = np.random.randint(4, 10)
            cv2.line(rgb, pt1, pt2, (60, 60, 60), road_width)
            # Roads absorb solar radiation, relatively warm in IR (medium gray)
            cv2.line(ir, pt1, pt2, np.random.randint(100, 130), road_width)

            # Add vehicles on road
            if np.random.rand() > 0.4:
                # Draw a tiny vehicle (hot spot in IR, colorful dot in RGB)
                t = np.random.rand()
                vx = int(pt1[0] * (1 - t) + pt2[0] * t)
                vy = int(pt1[1] * (1 - t) + pt2[1] * t)
                if 5 < vx < w - 5 and 5 < vy < h - 5:
                    # Vehicle in RGB (red, yellow, white, etc.)
                    colors = [(200, 20, 20), (220, 220, 0), (200, 200, 200)]
                    color = colors[np.random.randint(len(colors))]
                    cv2.rectangle(rgb, (vx - 2, vy - 2), (vx + 2, vy + 2), color, -1)
                    # Vehicle in IR: engine is very hot (bright white spot)
                    cv2.rectangle(ir, (vx - 2, vy - 2), (vx + 2, vy + 2), np.random.randint(230, 255), -1)

        # Add buildings (gray/brown blocks in RGB, warm in IR)
        num_buildings = np.random.randint(3, 10)
        for _ in range(num_buildings):
            bx, by = np.random.randint(10, w - 40), np.random.randint(10, h - 40)
            bw, bh = np.random.randint(15, 35), np.random.randint(15, 35)
            # RGB buildings
            b_color = (np.random.randint(100, 180), np.random.randint(100, 150), np.random.randint(100, 150))
            cv2.rectangle(rgb, (bx, by), (bx + bw, by + bh), b_color, -1)
            # IR buildings (typically warm concrete)
            ir_val = np.random.randint(120, 160)
            cv2.rectangle(ir, (bx, by), (bx + bw, by + bh), int(ir_val), -1)
            # Add roof details
            if np.random.rand() > 0.5:
                cv2.rectangle(rgb, (bx + 3, by + 3), (bx + bw - 3, by + bh - 3), (b_color[0]+20, b_color[1]+20, b_color[2]+20), -1)
                cv2.rectangle(ir, (bx + 3, by + 3), (bx + bw - 3, by + bh - 3), int(ir_val + np.random.choice([-15, 15])), -1)

        # Add sensor noise / blur to IR to make colorization realistic/challenging
        noise = np.random.normal(0, np.random.uniform(2, 6), ir.shape).astype(np.int16)
        ir_noisy = np.clip(ir.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        ir_noisy = cv2.GaussianBlur(ir_noisy, (3, 3), 0)

        # Save images
        cv2.imwrite(os.path.join(save_dir, "rgb", f"img_{idx:04d}.png"), rgb)
        cv2.imwrite(os.path.join(save_dir, "ir", f"img_{idx:04d}.png"), ir_noisy)

    print("Data generation complete.")

class SatelliteDataset(Dataset):
    def __init__(self, data_dir, image_size=256, is_train=True):
        self.data_dir = data_dir
        self.image_size = image_size
        self.is_train = is_train
        
        self.rgb_dir = os.path.join(data_dir, "rgb")
        self.ir_dir = os.path.join(data_dir, "ir")
        
        self.filenames = [f for f in os.listdir(self.ir_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Standard normalization: range [-1, 1] for GAN stability
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize((0.5,), (0.5,))
        self.normalize_rgb = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        ir_path = os.path.join(self.ir_dir, filename)
        rgb_path = os.path.join(self.rgb_dir, filename)
        
        ir_img = Image.open(ir_path).convert('L')
        rgb_img = Image.open(rgb_path).convert('RGB')
        
        # Resize
        ir_img = ir_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        rgb_img = rgb_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        # Data Augmentation (synchronized for pairs)
        if self.is_train:
            # Random Horizontal Flip
            if np.random.rand() > 0.5:
                ir_img = ir_img.transpose(Image.FLIP_LEFT_RIGHT)
                rgb_img = rgb_img.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Random Vertical Flip
            if np.random.rand() > 0.5:
                ir_img = ir_img.transpose(Image.FLIP_TOP_BOTTOM)
                rgb_img = rgb_img.transpose(Image.FLIP_TOP_BOTTOM)
                
            # Random Rotation (90 degrees)
            if np.random.rand() > 0.5:
                angle = np.random.choice([Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270])
                ir_img = ir_img.transpose(angle)
                rgb_img = rgb_img.transpose(angle)
        
        # Transform to tensor
        ir_tensor = self.to_tensor(ir_img)
        rgb_tensor = self.to_tensor(rgb_img)
        
        # Normalize
        ir_tensor = self.normalize(ir_tensor)
        rgb_tensor = self.normalize_rgb(rgb_tensor)
        
        # We also want to generate an "enhanced" target.
        # Since we simulate enhancement, we can define the enhancement target as a cleaned up, 
        # higher-contrast grayscale image. In real application, we can use CLAHE on the original IR or 
        # match a high-resolution panchromatic band.
        # Let's create an enhanced target by applying CLAHE to the original IR (as a simulated target).
        ir_np = np.array(ir_img)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_np = clahe.apply(ir_np)
        
        # Add a tiny bit of sharpening
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        enhanced_np = cv2.filter2D(enhanced_np, -1, kernel)
        
        enhanced_img = Image.fromarray(enhanced_np)
        enhanced_tensor = self.to_tensor(enhanced_img)
        enhanced_tensor = self.normalize(enhanced_tensor)
        
        return ir_tensor, enhanced_tensor, rgb_tensor

if __name__ == "__main__":
    # Test script to generate sample dataset
    generate_synthetic_data("data/train", num_images=10)
    dataset = SatelliteDataset("data/train", is_train=True)
    ir, enh, rgb = dataset[0]
    print("Dataset loaded successfully!")
    print(f"IR shape: {ir.shape}, Enhanced shape: {enh.shape}, RGB shape: {rgb.shape}")
