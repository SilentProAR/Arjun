import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import os

# UNet with BatchNorm + Dropout (same as training notebook)
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.2),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.down1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.up4 = conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up1 = conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)

        bn = self.bottleneck(p4)

        up4 = self.upconv4(bn)
        up4 = torch.cat([up4, d4], dim=1)
        up4 = self.up4(up4)

        up3 = self.upconv3(up4)
        up3 = torch.cat([up3, d3], dim=1)
        up3 = self.up3(up3)

        up2 = self.upconv2(up3)
        up2 = torch.cat([up2, d2], dim=1)
        up2 = self.up2(up2)

        up1 = self.upconv1(up2)
        up1 = torch.cat([up1, d1], dim=1)
        up1 = self.up1(up1)

        return self.final_conv(up1)

# Streamlit app setup
st.set_page_config(page_title="Brain Tumor Segmentation", layout="centered")
st.title("🧠 Brain Tumor Segmentation")

st.markdown("""
    <style>
        .image-container {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 20px;
        }
        .image-container img {
            border-radius: 15px;
            border: 2px solid #555;
            width: 300px;
            height: auto;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = UNet()
    model.load_state_dict(torch.load("unet_improved.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # Resize to 256x256
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

    # Normalize and Denoise
    image = image.astype(np.float32) / 255.0
    image_uint8 = (image * 255).astype(np.uint8)
    image_denoised = cv2.fastNlMeansDenoising(image_uint8, None, h=10, templateWindowSize=7, searchWindowSize=21)
    image_denoised = image_denoised.astype(np.float32) / 255.0

    # Convert to tensor
    tensor = torch.tensor(image_denoised, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return image_denoised, tensor

def predict_mask(model, tensor):
    with torch.no_grad():
        output = model(tensor)
        output = torch.sigmoid(output)
        output = (output > 0.5).float()
    return output.squeeze().cpu().numpy()

# File uploader
uploaded_file = st.file_uploader("Upload a brain image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    model = load_model()
    preprocessed_img, tensor = preprocess_image(uploaded_file)
    prediction = predict_mask(model, tensor)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Input Image**")
        st.image(preprocessed_img, channels="GRAY", use_column_width=True)
    with col2:
        st.markdown("**Predicted Mask**")
        st.image(prediction, channels="GRAY", use_column_width=True)
