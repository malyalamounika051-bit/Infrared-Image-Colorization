import os
import streamlit as st
from PIL import Image
import torch
import numpy as np
import io

# Import local modules
from predict import InferencePipeline
from dataset import generate_synthetic_data

st.set_page_config(
    page_title="Satellite IR Image Colorizer & Enhancer",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    .reportview-container {
        background: #0f111a;
    }
    h1 {
        color: #e0e6ed;
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .stButton>button {
        background-color: #6366f1;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #4f46e5;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
        transform: translateY(-1px);
    }
    .card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #334155;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #38bdf8;
    }
</style>
""", unsafe_allow_html=True)

# Main Title & Subtitle
st.title("🛰️ Satellite Infrared Colorization & Enhancement")
st.markdown("Convert monochrome infrared satellite imagery to high-contrast enhanced grayscale and realistic semantic RGB colorized outputs.")

# Directory Configuration
CHECKPOINT_PATH = "checkpoints/netG_final.pth"
SAMPLE_IMGS_DIR = "data/train/ir"

# Load Pipeline
@st.cache_resource
def get_pipeline(checkpoint_path):
    if os.path.exists(checkpoint_path):
        return InferencePipeline(checkpoint_path)
    return None

pipeline = get_pipeline(CHECKPOINT_PATH)

# Sidebar
st.sidebar.image("https://img.icons8.com/color/144/satellite.png", width=100)
st.sidebar.header("🔧 Configuration & Controls")

# Mode Selection
app_mode = st.sidebar.selectbox("Choose Mode", ["Single Image Inference", "Batch Inference", "Synthetic Data Generator & Quick Train"])

if app_mode == "Single Image Inference":
    st.sidebar.markdown("### Model Status")
    if pipeline is not None:
        st.sidebar.success("✅ Model checkpoint loaded successfully!")
    else:
        st.sidebar.warning("⚠️ No model checkpoint found. Go to 'Synthetic Data Generator & Quick Train' mode to generate data and train a quick checkpoint.")

    st.markdown("### 📤 Upload Infrared (IR) Satellite Image")
    
    col_upload, col_demo = st.columns([2, 1])
    
    uploaded_file = None
    with col_upload:
        uploaded_file = st.file_uploader("Choose an IR image...", type=["png", "jpg", "jpeg", "tif", "tiff"])
        
    # Quick demo image option if available
    selected_demo_img = None
    with col_demo:
        if os.path.exists(SAMPLE_IMGS_DIR) and len(os.listdir(SAMPLE_IMGS_DIR)) > 0:
            st.markdown("**Or pick a synthetic sample:**")
            sample_files = [f for f in os.listdir(SAMPLE_IMGS_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))][:5]
            selected_demo_img = st.selectbox("Sample image", ["None"] + sample_files)
            
    # Load selected image
    img_to_process = None
    if uploaded_file is not None:
        img_to_process = Image.open(uploaded_file)
    elif selected_demo_img and selected_demo_img != "None":
        img_to_process = Image.open(os.path.join(SAMPLE_IMGS_DIR, selected_demo_img))

    if img_to_process is not None:
        st.markdown("---")
        
        # Display side-by-side processing
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📷 Original IR Image")
            st.image(img_to_process, use_column_width=True)
            
        # Inference Action
        if pipeline is None:
            st.error("No model checkpoint found! Please train a model first using the 'Synthetic Data Generator & Quick Train' mode in the sidebar.")
        else:
            with st.spinner("Processing image via Dual-Branch Enhancement & Colorization pipeline..."):
                orig, enh, col = pipeline.predict_image(img_to_process)
                
            with col2:
                st.subheader("✨ Enhanced Grayscale")
                st.image(enh, use_column_width=True)
                # Download button
                buf = io.BytesIO()
                enh.save(buf, format="PNG")
                st.download_button(
                    label="📥 Download Enhanced Image",
                    data=buf.getvalue(),
                    file_name="enhanced_ir.png",
                    mime="image/png"
                )
                
            with col3:
                st.subheader("🎨 Realistic RGB Output")
                st.image(col, use_column_width=True)
                # Download button
                buf_col = io.BytesIO()
                col.save(buf_col, format="PNG")
                st.download_button(
                    label="📥 Download Colorized Image",
                    data=buf_col.getvalue(),
                    file_name="colorized_rgb.png",
                    mime="image/png"
                )
                
            # Side-by-side comparison grid
            st.markdown("### 🔍 Full Comparison Grid")
            w, h = orig.size
            grid = Image.new('RGB', (w * 3, h))
            grid.paste(orig.convert('RGB'), (0, 0))
            grid.paste(enh.convert('RGB'), (w, 0))
            grid.paste(col, (w * 2, 0))
            st.image(grid, use_column_width=True, caption="Original IR | Enhanced Grayscale | Colorized RGB")
            
            # Download Grid
            buf_grid = io.BytesIO()
            grid.save(buf_grid, format="PNG")
            st.download_button(
                label="📥 Download Complete Grid",
                data=buf_grid.getvalue(),
                file_name="comparison_grid.png",
                mime="image/png"
            )

elif app_mode == "Batch Inference":
    st.markdown("### 🗂️ Batch Inference Mode")
    st.markdown("Run enhancement and colorization on entire directories of satellite imagery.")
    
    input_folder = st.text_input("Input Directory Path", value="data/train/ir")
    output_folder = st.text_input("Output Directory Path", value="outputs/batch_results")
    
    if st.button("🚀 Run Batch Processing"):
        if pipeline is None:
            st.error("No model checkpoint found! Make sure checkpoints/netG_final.pth exists.")
        elif not os.path.exists(input_folder):
            st.error(f"Input directory '{input_folder}' does not exist.")
        else:
            with st.spinner("Processing batch..."):
                from predict import run_batch_inference
                run_batch_inference(CHECKPOINT_PATH, input_folder, output_folder, pipeline.device)
                st.success(f"Batch inference complete! Outputs saved to {output_folder}")

elif app_mode == "Synthetic Data Generator & Quick Train":
    st.markdown("### 🧠 Synthetic Data Generator & Quick Train Dashboard")
    st.markdown("Since satellite datasets are large and complex to retrieve directly, you can generate synthetic pairs and train a model checkpoint instantly to see the network learn mapping in real time.")
    
    col_gen, col_train = st.columns(2)
    
    with col_gen:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🎨 Generate Synthetic Dataset")
        num_images = st.slider("Number of image pairs", min_value=10, max_value=200, value=80, step=10)
        
        if st.button("Generate Synthetic Pairs"):
            with st.spinner("Generating satellite-like patterns..."):
                generate_synthetic_data("data/train", num_images=num_images)
                st.success(f"Successfully generated {num_images} RGB/IR pairs in 'data/train/'!")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_train:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🏋️ Run Quick Training Prototype")
        epochs = st.slider("Training Epochs", min_value=1, max_value=20, value=5)
        batch_size = st.selectbox("Batch Size", [4, 8, 16], index=1)
        
        if st.button("Start Training"):
            st.info("Starting training loop... progress will print in the backend terminal. Checkpoint will be written to 'checkpoints/netG_final.pth'.")
            # Running train.py via a command line simulation / subprocess or import
            with st.spinner("Training model... Check the system terminal console for training metrics."):
                import subprocess
                cmd = f"python train.py --epochs {epochs} --batch_size {batch_size} --data_dir data --checkpoint_dir checkpoints --samples_dir samples"
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                # Check for output checkpoint
                if os.path.exists("checkpoints/netG_final.pth"):
                    st.success("🎉 Checkpoint trained and successfully generated at checkpoints/netG_final.pth!")
                else:
                    # Copy epoch checkpoint if final isn't created due to early stop
                    ckpts = [f for f in os.listdir("checkpoints") if f.startswith("netG_epoch_")]
                    if len(ckpts) > 0:
                        import shutil
                        shutil.copy(os.path.join("checkpoints", ckpts[-1]), "checkpoints/netG_final.pth")
                        st.success("🎉 Checked model saved checkpoint as netG_final.pth!")
                    else:
                        st.error("Training failed to output a checkpoint. Output logs:")
                        st.code(res.stderr)
        st.markdown("</div>", unsafe_allow_html=True)
