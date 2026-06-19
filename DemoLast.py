import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models


#Load the model

PATH = "resnet18.pth"
num_classes = 32
loaded_state_dict = torch.load(PATH)
model_res = models.resnet18(pretrained=True)
model_res.fc = nn.Linear(model_res.fc.in_features, num_classes)
model_res.load_state_dict(loaded_state_dict)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_res = model_res.to(device).eval()
tags = [
    "ain (ع)", "al (ا)", "aleff (أ)", "bb (ب)", "dal (د)", "dha (ظ)", "dhad (ض)",
    "fa (ف)", "gaaf (ق)", "ghain (غ)", "ha (ه)", "haa (ح)", "jeem (ج)", "kaaf (ك)",
    "khaa (خ)", "la (ل)", "laam (ل)", "meem (م)", "nun (ن)", "ra (ر)", "saad (ص)",
    "seen (س)", "sheen (ش)", "ta (ت)", "taa (ط)", "thaa (ث)", "thal (ذ)", "toot (تـ)",
    "waw (و)", "ya (ي)", "yaa (ى)", "zay (ز)"
]


PATH = "MobileNet.pth"
loaded_state_dict = torch.load(PATH)
model_mobilenet = models.mobilenet_v3_small(pretrained=True)
in_features = model_mobilenet.classifier[3].in_features
model_mobilenet.classifier[3] = nn.Linear(in_features, num_classes)
model_mobilenet.load_state_dict(loaded_state_dict)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_mobilenet = model_mobilenet.to(device).eval()


PATH = "EfficientNet.pth"
loaded_state_dict = torch.load(PATH)
model_EfficientNet = models.efficientnet_b0(pretrained=True)
in_features = model_EfficientNet.classifier[1].in_features
model_EfficientNet.classifier[1] = nn.Linear(in_features, num_classes)
model_EfficientNet.load_state_dict(loaded_state_dict)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_EfficientNet = model_EfficientNet.to(device).eval()




transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a consistent size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize
    # Apply Augmentations to make the model used to different situations
])




import io
from PIL import Image
import streamlit as st

st.set_page_config(page_title="Image Prediction Demo", page_icon="📷", layout="centered")

st.title("📷 Sign Language Prediction")
st.write("Upload an image and see the model’s prediction.")

# ----------------------------
# Upload UI
# ----------------------------
st.subheader("1) Upload an image") 	
uploaded = st.file_uploader(
    "Choose a JPG image",
    type=["jpg"],
    accept_multiple_files=False
)

if uploaded is not None:
    # Display uploaded image
    image_bytes = uploaded.read()
    image = Image.open(io.BytesIO(image_bytes))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.subheader("2) Model prediction")
    img = image.copy()
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    outputs = model_res(img)
    _, prediction = torch.max(outputs.data, 1)
    st.write("Resnet18 Prediction:", tags[prediction[0]])


    img = image.copy()
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    outputs = model_mobilenet(img)
    _, prediction = torch.max(outputs.data, 1)
    st.write("MobileNetV3 Prediction", tags[prediction[0]])

    img = image.copy()
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    outputs = model_mobilenet(img)
    _, prediction = torch.max(outputs.data, 1)
    st.write("EfficentNet_B0 Prediction", tags[prediction[0]])

else:
    st.info("Upload an image to get started.")

