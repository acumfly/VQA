import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import silence_tensorflow.auto
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, LSTM, Embedding, Dense, Concatenate, Bidirectional, Dropout, BatchNormalization
from transformers import BlipForQuestionAnswering
import torch

tf.get_logger().setLevel('ERROR')

labels_dict = {'abdomen': 0, 'adrenal': 1, 'blood': 2, 'bone marrow': 3,
               'cardiovascular': 4, 'endocrine': 5, 'female reproductive': 6,
               'gastrointestinal': 7, 'hematologic': 8, 'hepatobiliary': 9,
               'joints': 10, 'liver': 11, 'lymph node': 12, 'nervous': 13, 'oral': 14,
               'pituitary': 15, 'respiratory': 16, 'soft tissue': 17, 'spleen': 18,
               'thymus': 19, 'uterus': 20, 'vasculature': 21}

ques_type_dict = {'number': 0, 'other': 1, 'yes/no': 2}

gen_mod_dir = 'D:/diploma/kaggle_data/generation'
generation_model = BlipForQuestionAnswering.from_pretrained(gen_mod_dir, torch_dtype=torch.float16, use_safetensors=True)
# model_multitarget = keras.models.load_model('models/model_resnet (1).keras')
model_multitarget = keras.models.load_model('models/model_resnet_aug2.keras')
yes_no_model = keras.models.load_model('models/VQA_bin_exp.keras')
ques_model = keras.models.load_model('models/ques_model.keras')

def make_preds_multitarget(img): #,processed_label
    # Указываем, что модель должна работать на CPU
    with tf.device('/cpu:0'):
        predictions = model_multitarget.predict(img)
        prediction_ind = np.argmax(predictions)
        reversed_dict = {value: key for key, value in labels_dict.items()}
        pred_label = reversed_dict[prediction_ind]
        return pred_label

def make_preds_binary(image, sequence):
    with tf.device('/cpu:0'):
        prediction = yes_no_model.predict([image, sequence])
        if prediction > 0.5:
            return 'yes'
        else:
            return 'no'
        # return pred_label

def make_preds_ques_type(sequence):
    with tf.device('/cpu:0'):
        predictions = ques_model.predict([sequence])
        prediction_ind = np.argmax(predictions)
        reversed_dict = {value: key for key, value in ques_type_dict.items()}
        pred_label = reversed_dict[prediction_ind]
        return pred_label


def make_generation(inputs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generation_model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = generation_model.generate(pixel_values=inputs['pixel_values'],
                            input_ids=inputs['input_ids'])
    return outputs