from Image_captioner import *
import streamlit as st
from PIL import Image
import pandas as pd

model, tokenizer, idx2word = model_instance()


def load_image(image_file):
    img = Image.open(image_file)
    img = img.convert("RGB")
    img = img.resize((299, 299))
    return img


def UI():
    uploaded_file = st.file_uploader(
        "Choose a file", type=['png', 'jpg', 'jpeg', ])
    if uploaded_file is not None:
        input_img = load_image(uploaded_file)
        img_to_arr = np.array(input_img)
        tf_arr = tf.convert_to_tensor(img_to_arr, dtype=tf.float32)

        img = load_image_from_path(tf_arr)
        caption = generate_caption(model, tokenizer, idx2word, img)

        st.image(input_img, width=250)
        st.markdown(caption)


if __name__ == '__main__':
    st.set_page_config(
        page_title="Image Captioning",
        page_icon="3️⃣",
    )
    UI()
