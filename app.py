import streamlit as st
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os

st.title("🧠 Brain Tumor Segmentation with SegFormer")
st.markdown("Upload an MRI image to detect brain tumors using a fine-tuned SegFormer model.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
    
    # Set the classifier to match the training structure
    model.decode_head.classifier = torch.nn.Conv2d(256, 1, kernel_size=1)

    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

@st.cache_resource
def load_feature_extractor():
    return SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

model = load_model()
feature_extractor = load_feature_extractor()

uploaded_file = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        preds = torch.sigmoid(logits) > 0.5  # Binary threshold

    mask = preds.squeeze().cpu().numpy().astype(np.uint8) * 255
    mask_img = Image.fromarray(mask)

    st.subheader("🧪 Predicted Tumor Segmentation")
    st.image(mask_img, caption="Segmentation Mask", use_column_width=True)
