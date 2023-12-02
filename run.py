import streamlit as st
import numpy as np
from PIL import Image
import torch
import io

# Load the pre-trained model
model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v2", trust_repo=True)
face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", size=512, trust_repo=True)


def main():
    st.title("AnimeGAN2 Face Paint App")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Process the image and display the result
        if st.button("Generate Anime Face Paint"):
            result_image = generate_face_paint(image)
            st.image(result_image, caption="Generated Image", use_column_width=True)

            # Save the PIL Image to a BytesIO object
            result_image_bytes = image_to_bytes(result_image)

            # Add a download button
            download_button = st.download_button(
                label="Download Result Image",
                data=result_image_bytes,
                file_name="anime_face_paint.png",
                key="download_button"
            )

def generate_face_paint(image):
    # Apply face paint using the pre-trained models
    with st.spinner("Generating..."):
        out = face2paint(model, image)

    # No need for .cpu() or .numpy(), directly use the PIL Image
    result_image = out

    return result_image


def image_to_bytes(image):
    # Convert a PIL Image to BytesIO object
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format='PNG')
    return img_byte_array.getvalue()

if __name__ == "__main__":
    main()

