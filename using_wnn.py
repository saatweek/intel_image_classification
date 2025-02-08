# All the imports
from tensorflow.keras.models import Model
import os
import kagglehub
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import ao_arch as ar
import ao_core as ao

# just a sample image of a building for testing [remove later]
img = "C:/Users/satwi/.cache/kagglehub/datasets/puneet6060/intel-image-classification/versions/2/seg_train/seg_train/buildings/0.jpg"


# Download data and set the training, validation and prediction directories
path = kagglehub.dataset_download("puneet6060/intel-image-classification")
train_dir = os.path.join(path, os.path.join("seg_train", "seg_train"))
validation_dir = os.path.join(path, os.path.join("seg_test", "seg_test"))
pred_dir = os.path.join(path, os.path.join("seg_pred", "seg_pred"))


# Loading and specifying the intermediate model
model = tf.keras.models.load_model("./convolution.keras")
layer_name = 'dense'
intermediate_layer_model = Model(inputs=model.inputs, outputs=model.get_layer(layer_name).output)

#Image Transformer
def transform_image(img_path, target_size=(150, 150, 3)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array /= 255 # normalization
    # img_array = np.expand_dims(img_array, 0) ## uncomment when predicting only a single image ##

    return img_array

# Assigning an integer to each label
label_dict= {}
for idx, item in enumerate(os.listdir(train_dir)):
    label_dict[idx] = item

# Taking all the inputs and their labels
input_arr = []
label_arr = []
for items in label_dict.keys():
    temp_path = os.path.join(train_dir, label_dict[items])
    for pic in os.listdir(temp_path):
        input_arr.append(transform_image(os.path.join(temp_path, pic)))
        label_arr.append(items)

input_arr = np.asarray(input_arr)
print(input_arr.shape)
# weightless neural network architecture
arch_i = [256]
arch_z = [6]
arch_c = []
connector_function = "rand_conn"
description = "On top of CNN"
connector_parameters = [256, 200, 256, 6]
arch = ar.Arch(arch_i, arch_z, arch_c, connector_function, connector_parameters, description)

# Image pipeline from convolution to wnn
# inter_output = []
# for items in input_arr:
#     inter_output.append(intermediate_layer_model.predict(items)[0])

# print(inter_output[:2 ])
# print(len(inter_output))
# feature_output = intermediate_layer_model.predict(img_array)
# agent = ao.Agent(arch, notes="On top of CNN", save_meta=False)

# for i in range(len(inter_output)):
#     agent.next_state(inter_output[i], [label_arr[i]], DD=False, unsequenced=False)
# agent.next_state_batch(inter_output, label_arr, DD=False, unsequenced=True)

# agent.pickle()


#Experimenting with batch processing
inter_output = intermediate_layer_model.predict(input_arr)
print(inter_output.shape)