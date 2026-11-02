"""Hand gesture recognition web app (Assignment 2).

Upload a photo or use the webcam -> MobileNetV4 predicts 1 / 2 / 3 fingers
-> Grad-CAM shows which pixels the decision is based on.

Run:  streamlit run app/app.py
"""
import os
import platform
import time

import numpy as np
import streamlit as st
import timm
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from matplotlib import colormaps
from torchvision import transforms as T

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'mobilenetv4_gesture.pt')

st.set_page_config(page_title="Hand Gesture Recognition", page_icon="", layout="wide")


class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        layer.register_forward_hook(self._save)

    def _save(self, module, inputs, output):
        self.activations = output
        if output.requires_grad: 
            output.register_hook(lambda grad: setattr(self, 'gradients', grad))

    def __call__(self, x):
        logits = self.model(x)
        class_idx = logits.argmax(1).item()
        self.model.zero_grad()
        logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().numpy(), F.softmax(logits, 1)[0].detach().numpy()


@st.cache_resource
def load_model():
    ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model = timm.create_model(ckpt['model_name'], pretrained=False, num_classes=len(ckpt['class_names']))
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    preprocess = T.Compose([
        T.Resize((ckpt['img_size'], ckpt['img_size'])),
        T.ToTensor(),
        T.Normalize(ckpt['mean'], ckpt['std']),
    ])
    with torch.no_grad():  
        for _ in range(3):
            model(torch.zeros(1, 3, ckpt['img_size'], ckpt['img_size']))
    return model, GradCAM(model, model.blocks[-1]), preprocess, ckpt


def overlay(img, cam, alpha=0.45):
    heat = (colormaps['jet'](cam)[..., :3] * 255).astype('uint8')
    heat = Image.fromarray(heat).resize(img.size, Image.BILINEAR)
    return Image.blend(img, heat, alpha)


def browser_name():
    try:
        ua = st.context.headers.get('User-Agent', '')
    except Exception:
        return 'unknown'
    for name in ['Edg', 'OPR', 'Chrome', 'Firefox', 'Safari']:
        if name in ua:
            return {'Edg': 'Edge', 'OPR': 'Opera'}.get(name, name)
    return ua[:40] or 'unknown'


if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: `{MODEL_PATH}`.\n\n"
             "Run the Colab notebook, download `outputs/mobilenetv4_gesture.pt` and put it in the `app/` folder.")
    st.stop()

model, gradcam, preprocess, ckpt = load_model()
labels = ckpt['class_labels']

if 'latencies' not in st.session_state:
    st.session_state.latencies = []

# sidebar: model and environment 
with st.sidebar:
    st.header("Model")
    st.write(f"**MobileNetV4** (`{ckpt['model_name'].split('.')[0]}`)")
    st.write(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    m = ckpt.get('test_metrics', {})
    if m:
        st.write(f"Test accuracy: **{m['Accuracy']:.1%}** · macro-F1: **{m['F1 (macro)']:.3f}**")

    st.header("Environment")
    st.write(f"Device: **CPU** ({platform.processor() or platform.machine()})")
    st.write(f"OS: {platform.system()} {platform.release()}")
    st.write(f"Browser: {browser_name()}")
    st.write(f"PyTorch {torch.__version__} · {torch.get_num_threads()} threads")

    st.header("Latency")
    latency_box = st.empty()  # filled in after the prediction below
    latency_box.write("No predictions yet.")

    st.divider()
    st.caption("Privacy: images are processed in memory and are not saved. "
               "Only use photos of people who agreed to it.")

# main page 
st.title("Hand Gesture Recognition")
st.write("Show **one, two or three fingers** (palm facing the camera). "
         "The model predicts the gesture and Grad-CAM highlights the pixels it looked at.")

tab_upload, tab_camera = st.tabs(["Upload photo", "Webcam"])
with tab_upload:
    uploaded = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
with tab_camera:
    captured = st.camera_input("Take a photo")

source = captured or uploaded
if source is None:
    st.info("Upload a photo or take one with the webcam to get a prediction.")
    st.stop()

img = ImageOps.exif_transpose(Image.open(source)).convert('RGB')
x = preprocess(img).unsqueeze(0)

t0 = time.perf_counter()
with torch.no_grad():
    probs = F.softmax(model(x), 1)[0].numpy()
latency = (time.perf_counter() - t0) * 1000
st.session_state.latencies.append(latency)
lat = st.session_state.latencies
latency_box.metric("Average inference", f"{np.mean(lat):.1f} ms", help=f"over {len(lat)} predictions")

cam, _ = gradcam(x)
pred = int(probs.argmax())

col_img, col_cam, col_pred = st.columns([1, 1, 1])
with col_img:
    st.subheader("Input")
    st.image(img, use_container_width=True)
with col_cam:
    st.subheader("Grad-CAM")
    st.image(overlay(img, cam), use_container_width=True)
    st.caption("Red = regions that increased the score of the predicted class.")
with col_pred:
    st.subheader("Prediction")
    st.markdown(f"## {labels[pred]}")
    st.write(f"Confidence: **{probs[pred]:.1%}**")
    for label, p in zip(labels, probs):
        st.progress(float(p), text=f"{label}: {p:.1%}")
    st.write(f"Inference time: **{latency:.1f} ms** (CPU)")
    if probs[pred] < 0.6:
        st.warning("Low confidence. Try a plain background, good light and the hand in the centre.")
