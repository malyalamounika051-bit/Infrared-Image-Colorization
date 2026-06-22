import os
import argparse
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2

# Import local modules
from models import DualBranchUNet

class InferencePipeline:
    def __init__(self, checkpoint_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.netG = DualBranchUNet(n_channels=1)
        self.netG.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.netG.to(device)
        self.netG.eval()
        
        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])

    def predict_image(self, img_path):
        """
        Loads, preprocesses, runs inference, and returns PIL images for
        (input_ir, enhanced_grayscale, colorized_rgb).
        """
        if isinstance(img_path, str):
            orig_img = Image.open(img_path).convert('L')
        else:
            # Assume PIL image or file-like object passed directly (Streamlit)
            orig_img = img_path.convert('L')
            
        w, h = orig_img.size
        
        # Preprocess
        tensor_in = self.transform(orig_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            enh_out, col_out = self.netG(tensor_in)
            
        # Post-process Enhancement output
        enh_out = (enh_out.squeeze(0).cpu() + 1.0) / 2.0
        enh_out = torch.clamp(enh_out, 0, 1)
        enh_pil = T.ToPILImage()(enh_out).resize((w, h), Image.BICUBIC)
        
        # Post-process Colorization output
        col_out = (col_out.squeeze(0).cpu() + 1.0) / 2.0
        col_out = torch.clamp(col_out, 0, 1)
        col_pil = T.ToPILImage()(col_out).resize((w, h), Image.BICUBIC)
        
        return orig_img, enh_pil, col_pil

def run_batch_inference(checkpoint_path, input_folder, output_folder, device):
    os.makedirs(output_folder, exist_ok=True)
    pipeline = InferencePipeline(checkpoint_path, device)
    
    supported_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(supported_extensions)]
    
    if len(files) == 0:
        print(f"No images found in {input_folder}")
        return
        
    print(f"Running batch inference on {len(files)} images...")
    for f in files:
        img_path = os.path.join(input_folder, f)
        orig, enh, col = pipeline.predict_image(img_path)
        
        # Save enhancement
        enh_name = f"enhanced_{f}"
        enh.save(os.path.join(output_folder, enh_name))
        
        # Save colorized
        col_name = f"colorized_{f}"
        col.save(os.path.join(output_folder, col_name))
        
        # Save side-by-side grid
        grid_name = f"grid_{f}"
        w, h = orig.size
        grid = Image.new('RGB', (w * 3, h))
        grid.paste(orig.convert('RGB'), (0, 0))
        grid.paste(enh.convert('RGB'), (w, 0))
        grid.paste(col, (w * 2, 0))
        grid.save(os.path.join(output_folder, grid_name))
        
    print(f"Batch inference complete. Outputs saved to {output_folder}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/netG_final.pth", help="Path to generator model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Path to input IR image or folder")
    parser.add_argument("--output", type=str, default="outputs", help="Directory or file path to save results")
    parser.add_argument("--batch", action="store_true", help="Set to run batch inference if input is a directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    args = parser.parse_args()

    if args.batch:
        run_batch_inference(args.checkpoint, args.input, args.output, args.device)
    else:
        pipeline = InferencePipeline(args.checkpoint, args.device)
        orig, enh, col = pipeline.predict_image(args.input)
        
        os.makedirs(args.output, exist_ok=True)
        base_name = os.path.basename(args.input)
        
        orig.save(os.path.join(args.output, f"original_{base_name}"))
        enh.save(os.path.join(args.output, f"enhanced_{base_name}"))
        col.save(os.path.join(args.output, f"colorized_{base_name}"))
        
        # Create and save grid
        w, h = orig.size
        grid = Image.new('RGB', (w * 3, h))
        grid.paste(orig.convert('RGB'), (0, 0))
        grid.paste(enh.convert('RGB'), (w, 0))
        grid.paste(col, (w * 2, 0))
        grid.save(os.path.join(args.output, f"grid_{base_name}"))
        print(f"Results saved to directory: {args.output}")

if __name__ == "__main__":
    main()
