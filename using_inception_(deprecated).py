import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import plotly.graph_objects as go
from data_prep import train_dir, validation_dir, cnn_preprocessing
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Flatten,Dense,Dropout
from tensorflow.keras import Model


pre_trained_model = InceptionV3(input_shape=(150,150,3),include_top=False,weights='imagenet')
# Get 'mixed9' layer as the last layer
last_layer = pre_trained_model.get_layer('mixed9_1')
last_output = last_layer.output
x = Flatten()(last_output)
x = Dense(units=1024,activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(units=256, activation='relu')(x)
x = Dense(units=6,activation='softmax')(x)
model = Model(pre_trained_model.input,x)

#get the summary
model.summary()

# train_datagen = ImageDataGenerator(rescale = 1./255, rotation_range=0.3, width_shift_range=0.3,
#     height_shift_range=0.3,shear_range=0.3, zoom_range=0.3, horizontal_flip=True)
#
# validation_datagen = ImageDataGenerator(rescale = 1./255, rotation_range=0.3, width_shift_range=0.3,
#     height_shift_range=0.3,shear_range=0.3, zoom_range=0.3, horizontal_flip=True)
#
# train_generator = train_datagen.flow_from_directory(train_dir, target_size = (150, 150), batch_size = 128, class_mode='sparse')

# validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size = (150, 150), batch_size = 128, class_mode='sparse')

class CallBack(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if logs.get('acc')>0.98:
      print('stop traning')
      self.model.stop_training=True
callback = CallBack()
train_data, train_label = cnn_preprocessing(train_dir)
val_data, val_label = cnn_preprocessing(validation_dir)
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics = ['acc'])
num_epochs = 40
history = model.fit(train_data, train_label, steps_per_epoch=20, epochs=num_epochs, validation_data = (val_data, val_label), callbacks=[callback])
model.save('./inception.keras')
fig = go.Figure()
x = np.linspace(1,num_epochs, num_epochs)
y1 = history.history['acc']
y2 = history.history['val_acc']
fig.add_trace(go.Scatter(x = x, y = y1, name = 'training accuracy')),
fig.add_trace(go.Scatter(x = x, y = y2, name = 'validation accuracy')),
fig.update_layout(xaxis_title = 'Epochs', yaxis_title = 'accuracy', title = 'Accuracy of Model')
# fig.show()
fig.write_html("Plots/inception.html")