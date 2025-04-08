
# Brain Tumor Detection using Custom Trained SegFormer

This project demonstrates the use of a **custom-trained SegFormer model** for brain tumor detection from MRI scans using semantic segmentation. The solution includes a PyTorch training pipeline and a Streamlit dashboard for real-time predictions.

---

## 🔬 Custom Training Overview

- **Model Architecture**: `SegFormer` from Hugging Face Transformers (`nvidia/segformer-b0-finetuned-ade-512-512`)
- **Dataset**: Custom brain MRI images with segmentation masks
- **Training Details**:
  - Input Size: `256x256`
  - Loss: `BCEWithLogitsLoss` + IoU
  - Optimizer: `AdamW`
  - Learning Rate: 3e-5
  - Epochs: 100+
- **Metrics**:
  - IoU Score
  - Dice Coefficient
  - Loss

---

## 🧠 Features

- Custom model trained for medical segmentation tasks
- Handles grayscale and color MRI images
- Real-time prediction via Streamlit
- Model runs on CPU or GPU
- Easy to test with sample images

---

## 🛠 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Aritra960966/brain_tumor_detection_withcustom_trained_segformer.git
cd brain_tumor_detection_withcustom_trained_segformer
