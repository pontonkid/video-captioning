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
