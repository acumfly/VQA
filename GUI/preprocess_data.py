import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import os
# Отключение предупреждений oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import re
import numpy as np
import cv2
from keras_preprocessing import image
import tensorflow as tf
from keras_applications.vgg16 import preprocess_input
from keras import backend as K, layers, models, utils
import pickle
from tensorflow import keras
from keras_preprocessing.sequence import pad_sequences
import pandas as pd
from PIL import Image
from transformers import BlipProcessor
# from keras_preprocessing.text import Tokenizer

# Установка формата данных изображения
K.set_image_data_format('channels_last')
multitarget_test = pd.read_csv('datasets/multitarget_test.csv')

with open('tokenizers/tokenizer_binary.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('tokenizers/tokenizer_questions.pkl', 'rb') as f:
    tokenizer_ques = pickle.load(f)
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

def preprocess_multitarget(img_path, question):
    # Загрузка и предварительная обработка изображения
    img = image.load_img(img_path, target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img, backend=K, layers=layers, models=models, utils=utils)
    label = multitarget_test[multitarget_test['image_path']==img_path].target
    return img, label

def preprocess_binary(img_path, question):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (256, 256))  # Изменение размера изображения
    image = image / 255.0  # Нормализация
    image = np.expand_dims(image, axis=0)  # Добавление измерения для батча

    sequence = tokenizer.texts_to_sequences([question])
    sequence = pad_sequences(sequence, maxlen=36, padding='post')

    return image, sequence

def preprocess_generation(img_path, question):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(image, question, return_tensors="pt")
    print(inputs)
    return inputs

def preprocess_ques_type(question):
    sequence = tokenizer_ques.texts_to_sequences([question])
    sequence = pad_sequences(sequence, maxlen=38, padding='post')
    return sequence

def decode_outputs(outputs):
    answer = processor.decode(outputs[0], skip_special_tokens=True)
    return answer