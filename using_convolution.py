import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import plotly.graph_objects as go
import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("puneet6060/intel-image-classification")

train_dir = os.path.join(path, os.path.join("seg_train", "seg_train"))
validation_dir = os.path.join(path, os.path.join("seg_test", "seg_test"))
pred_dir = os.path.join(path, os.path.join("seg_pred", "seg_pred"))

class myCallback (tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs = {}):
    if (logs.get('acc')>0.95):
      self.model.stop_training = True
      print('Enough Accuracy Reached!')

callback = myCallback()

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range=0.3,
                                   width_shift_range=0.3,
                                   height_shift_range=0.3,
                                   shear_range=0.3,
                                   zoom_range=0.3,
                                   horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale = 1./255,
                                        rotation_range=0.3,
                                        width_shift_range=0.3,
                                        height_shift_range=0.3,
                                        shear_range=0.3,
                                        zoom_range=0.3,
                                        horizontal_flip=True)
    
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size = (150, 150),
                                                    batch_size = 128,
                                                    class_mode='sparse')

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              target_size = (150, 150),
                                                              batch_size = 128,
                                                              class_mode='sparse')

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16, (3, 3), activation = 'relu', input_shape = (150, 150, 3)),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(256, activation='relu'),
                                    tf.keras.layers.Dense(6, activation = 'softmax')])

model.summary()

#hyperparamters
num_epochs = 40


model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer='adam',
              metrics = ['acc'])

history = model.fit(train_generator,
                    steps_per_epoch=20,
                    epochs=num_epochs,
                    validation_data = validation_generator,
                    callbacks = [callback])

model.save('./convolution.keras')
fig = go.Figure()
x = np.linspace(1, num_epochs, num_epochs)
y1 = history.history['acc']
y2 = history.history['val_acc']
fig.add_trace(go.Scatter(x = x, y = y1, name = 'training accuracy')),
fig.add_trace(go.Scatter(x = x, y = y2, name = 'validation accuracy')),
fig.update_layout(xaxis_title = 'Epochs', yaxis_title = 'accuracy', title = 'Accuracy of Model')
fig.show()