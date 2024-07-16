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
