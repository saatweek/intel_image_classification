# All the imports
from tensorflow.keras.models import Model
import os
import kagglehub
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import ao_arch as ar
import ao_core as ao
import struct

#Image Transformer
def transform_image(img_path, target_size=(150, 150, 3)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array /= 255 # normalization
    # img_array = np.expand_dims(img_array, 0) ## uncomment when predicting only a single image ##

    return img_array

#Image Preprocessing
def preprocessing(folder_path):
    # Assigning an integer to each label
    label_dict = {}
    for idx, item in enumerate(os.listdir(folder_path)):
        label_dict[idx] = item

    # Taking all the inputs and their labels
    input_arr = []
    label_arr = []
    for items in label_dict.keys():
        temp_path = os.path.join(folder_path, label_dict[items])
        for pic in os.listdir(temp_path):
            input_arr.append(transform_image(os.path.join(temp_path, pic)))
            label_arr.append(items)

    input_arr = np.asarray(input_arr)
    label_arr = np.asarray(label_arr)

    label_col = one_hot_encode(label_arr)

    return input_arr, label_col


def one_hot_encode(arr):
    unique_values = np.unique(arr)
    num_categories = len(unique_values)
    encoded_arr = np.zeros((len(arr), num_categories))

    for i, val in enumerate(arr):
        index = np.where(unique_values == val)[0][0]
        encoded_arr[i, index] = 1

    return encoded_arr

def binary(num):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))


if __name__=="__main__":
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

    # weightless neural network architecture
    arch_i = [256]
    arch_z = [6]
    arch_c = []
    connector_function = "rand_conn"
    description = "On top of CNN"
    connector_parameters = [256, 200, 256, 6]
    arch = ar.Arch(arch_i, arch_z, arch_c, connector_function, connector_parameters, description)

    # batch processing
    training_input, training_output = preprocessing(train_dir)
    inter_output = intermediate_layer_model.predict(training_input)
    print(inter_output.shape)
    print(training_output[:5])
    print(inter_output[:5])
    agent = ao.Agent(arch, notes="On top of CNN", save_meta=False)

    agent.next_state_batch(inter_output, training_output, DD=False, unsequenced=True)
    test_input, test_output = preprocessing(validation_dir)

    agent.pickle()


