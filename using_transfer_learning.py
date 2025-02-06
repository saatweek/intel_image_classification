import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import plotly.graph_objects as go
import tensorflow_hub as hub
import kagglehub
import os

file_path = kagglehub.dataset_download("puneet6060/intel-image-classification")
# model_path = kagglehub.model_download("tensorflow/inception/tfLite/v4")

train_dir = os.path.join(file_path, os.path.join("seg_train", "seg_train"))
validation_dir = os.path.join(file_path, os.path.join("seg_test", "seg_test"))

train_datagen = ImageDataGenerator(rescale = 1./255, rotation_range=0.3, width_shift_range=0.3,
    height_shift_range=0.3,shear_range=0.3, zoom_range=0.3, horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale = 1./255, rotation_range=0.3, width_shift_range=0.3,
    height_shift_range=0.3,shear_range=0.3, zoom_range=0.3, horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(train_dir, target_size = (150, 150), batch_size = 128, class_mode='sparse')

validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size = (150, 150), batch_size = 128, class_mode='sparse')

url = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4'

base_model = hub.KerasLayer(url, input_shape = (150, 150, 3))

model = tf.keras.Sequential([base_model,
                             tf.keras.layers.Dense(128, activation='relu'),
                             tf.keras.layers.Dense(64, activation='relu'),
                             tf.keras.layers.Dense(6, activation='softmax')])
model.summary()