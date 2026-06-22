# Satellite Infrared Image Colorization and Enhancement

An end-to-end deep learning system built in PyTorch to convert monochrome infrared (IR) satellite imagery into high-contrast enhanced grayscale images and realistic semantic RGB colorized outputs.

## 🚀 Features

- **Dual-Branch Architecture**: Integrated shared encoder with separate decoders for grayscale enhancement and RGB colorization.
- **Feature Fusion**: Incorporates feature fusion layers from the enhancement branch into the colorization branch to preserve structural boundaries (roads, water bodies, buildings).
- **Synthetic Data Generator**: Simulates satellite structures (roads, vegetation, buildings, water, vehicles) and their IR signatures to train the model end-to-end.
- **Interactive Streamlit App**: Upload custom images, visualize live metrics, run batch inference, and download processing outputs.

---

## 🛠️ Installation & Setup

1. **Install Dependencies**:
   ```bash
   pip install torch torchvision numpy opencv-python pillow streamlit tqdm
   ```

2. **Generate Dataset and Train (Quick Prototype)**:
   You can either run the dataset generator and train from the command line:
   ```bash
   # Generates 150 synthetic images and trains for 5 epochs (for rapid check)
   python train.py --epochs 5 --batch_size 8
   ```
   Or use the **Synthetic Data Generator & Quick Train** dashboard in the Streamlit application.

3. **Run Streamlit Web App**:
   ```bash
   streamlit run app.py
   ```

---

## 📂 Project Structure

- `models.py`: Dual-Branch U-Net Generator & PatchGAN Discriminator.
- `losses.py`: Reconstruction L1 loss, GAN Adversarial loss, and VGG-based Perceptual loss.
- `dataset.py`: Satellite dataset loader & synthetic data simulator.
- `train.py`: Model training loop and performance evaluator (PSNR, SSIM).
- `predict.py`: Inference utility for single/batch processing.
- `app.py`: Minimal web app user interface.
