import streamlit as st
import os
import cv2
import pandas as pd
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
import nltk
import tempfile
import zipfile
from nltk.corpus import wordnet
import spacy
import io
from spacy.cli import download


# Download necessary NLP models
nltk.download('wordnet')
nltk.download('omw-1.4')
download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm") 

# Load the pre-trained models for image captioning and summarization
model_name = "NourFakih/Vit-GPT2-COCO2017Flickr-85k-09"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPT2 only has bos/eos tokens but not decoder_start/pad tokens
tokenizer.pad_token = tokenizer.eos_token
# update the model config
model.config.eos_token_id = tokenizer.eos_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id


model_sum_name = "google-t5/t5-base"
tokenizer_sum = AutoTokenizer.from_pretrained("google-t5/t5-base")
model_sum = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")
summarize_pipe = pipeline("summarization", model=model_sum_name)

def generate_caption(image):
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption
    

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms


def preprocess_query(query):
    doc = nlp(query)
    tokens = set()
    for token in doc:
        tokens.add(token.text)
        tokens.add(token.lemma_)
        tokens.update(get_synonyms(token.text))
    return tokens

def search_captions(query, captions):
    query_tokens = preprocess_query(query)
    
    results = []
    for path, caption in captions.items():
        caption_tokens = preprocess_query(caption)
        if query_tokens & caption_tokens:
            results.append((path, caption))
    
    return results


def process_video(video_path, frame_interval):
    cap = cv2.VideoCapture(video_path)
    frames = []
    captions = []
    success, frame = cap.read()
    count = 0
    while success:
        if count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            caption = generate_caption(pil_image)
            frames.append(frame)
            captions.append(caption)
        success, frame = cap.read()
        count += 1
    cap.release()
    df = pd.DataFrame({'Frame': frames, 'Caption': captions})
    return frames, df

st.title("Combined Video Captioning and Gallery App")

# Sidebar for search functionality
with st.sidebar:
    query = st.text_input("Search videos by caption:")

# Options for input strategy
input_option = st.selectbox("Select input method:", ["Folder Path", "Upload Video", "Upload ZIP"])


video_files = []

if input_option == "Folder Path":
    folder_path = st.text_input("Enter the folder path containing videos:")
    if folder_path and os.path.isdir(folder_path):
        video_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('mp4', 'avi', 'mov', 'mkv'))]

elif input_option == "Upload Video":
    uploaded_files = st.file_uploader("Upload video files", type=["mp4", "avi", "mov", "mkv"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                video_files.append(temp_file.name)


elif input_option == "Upload ZIP":
    uploaded_zip = st.file_uploader("Upload a ZIP file containing videos", type=["zip"])
    if uploaded_zip:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_zip.read())
            with zipfile.ZipFile(temp_file.name, 'r') as zip_ref:
                zip_ref.extractall("/tmp/videos")
                video_files = [os.path.join("/tmp/videos", f) for f in zip_ref.namelist() if f.lower().endswith(('mp4', 'avi', 'mov', 'mkv'))]


if video_files:
    captions = {}
    for video_file in video_files:
        frames, captions_df = process_video(video_file, frame_interval=20)
        
        if frames and not captions_df.empty:
            generated_captions = ' '.join(captions_df['Caption'])
            summary = summarize_pipe(generated_captions)[0]['summary_text']
            captions[video_file] = summary


# Display videos in a 4-column grid
    cols = st.columns(4)
    for idx, (video_path, summary) in enumerate(captions.items()):
        with cols[idx % 4]:
            st.video(video_path)
            st.caption(summary)


if query:
        results = search_captions(query, captions)
        st.write("Search Results:")
        for video_path, summary in results:
            st.video(video_path)
            st.caption(summary)

# Save captions to CSV and provide a download button
    if st.button("Generate CSV"):
        df = pd.DataFrame(list(captions.items()), columns=['Video', 'Caption'])
        csv = df.to_csv(index=False)
        st.download_button(label="Download captions as CSV", data=csv, file_name="captions.csv", mime="text/csv")

