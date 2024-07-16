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


