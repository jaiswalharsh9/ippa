import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- App title ---
st.markdown("<h1 style='text-align: center;'>🖼️ Advanced Image Processing App</h1>", unsafe_allow_html=True)
st.markdown("### Upload an image and apply cool filters in real time!")

# --- Sidebar ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/OpenCV_Logo_with_text_svg_version.svg/512px-OpenCV_Logo_with_text_svg_version.svg.png", width=100)
st.sidebar.markdown("## 🛠️ Filters")
st.sidebar.markdown("Choose a filter to apply to your uploaded image.")

# --- Upload image ---
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

# --- Image Filters ---
def apply_sepia(img):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_img = cv2.transform(img, sepia_filter)
    return np.clip(sepia_img, 0, 255).astype(np.uint8)

def apply_sketch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return sketch

# --- Main processing ---
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    option = st.sidebar.selectbox(
        "🎨 Select a filter",
        ["Grayscale", "Canny Edge Detection", "Blur", "Sepia", "Invert Colors", "Sketch"]
    )

    # --- Filter processing ---
    if option == "Grayscale":
        processed = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    elif option == "Canny Edge Detection":
        low = st.sidebar.slider("Min Threshold", 0, 100, 50)
        high = st.sidebar.slider("Max Threshold", 100, 300, 150)
        processed = cv2.Canny(img_array, low, high)

    elif option == "Blur":
        k = st.sidebar.slider("Kernel Size", 1, 15, 5, step=2)
        processed = cv2.GaussianBlur(img_array, (k, k), 0)

    elif option == "Sepia":
        processed = apply_sepia(img_array)

    elif option == "Invert Colors":
        processed = cv2.bitwise_not(img_array)

    elif option == "Sketch":
        processed = apply_sketch(img_array)

    # --- Layout: side-by-side view ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🖼️ Original Image")
        st.image(image, use_column_width=True)

    with col2:
        st.markdown(f"#### ✨ {option} Image")
        if option in ["Grayscale", "Canny Edge Detection", "Sketch"]:
            st.image(processed, use_column_width=True, channels="GRAY")
        else:
            st.image(processed, use_column_width=True)

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center;'>Built with ❤️ using Streamlit & OpenCV</p>", unsafe_allow_html=True)

