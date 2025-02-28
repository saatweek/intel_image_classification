# All the imports
from typing import Any

from numpy import ndarray, dtype, floating
from numpy._typing import _64Bit
from tensorflow.keras.models import Model
import os
import kagglehub
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import ao_arch as ar
import ao_core as ao
from sklearn.utils import shuffle


#Image Transformer
def transform_image(img_path:str, target_size:tuple=(150, 150, 3)) -> np.ndarray:
    """
    :param img_path: str : Path of the image
    :param target_size: tuple : dimensions of the image (height, width and the color channels)
    :return: image in the form of a numpy array
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array /= 255 # normalization
    # img_array = np.expand_dims(img_array, 0) ## uncomment when predic ng only a single image ##

    return img_array

#Image Preprocessing
def preprocessing(folder_path:str, label_dictionary:dict) -> tuple[np.ndarray, np.ndarray]:
    """
    :param folder_path:  str : takes the path of the folder in which all the images are present
    :param label_dictionary: label dictionary containing the mapping of each label to an integer
    :return:  All the images (in the form of a numpy array) and their corresponding labels (2 numpy arrays)
    """
    # Taking all the inputs and their labels
    input_arr = []
    label_arr = []
    for items in label_dictionary.keys():
        temp_path = os.path.join(folder_path, label_dictionary[items])
        for pic in os.listdir(temp_path):
            input_arr.append(transform_image(os.path.join(temp_path, pic)))
            label_arr.append(items)

    input_arr = np.asarray(input_arr)
    label_arr = np.asarray(label_arr)

    input_arr, label_arr = shuffle(input_arr, label_arr)


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

# Download data and set the training, validation and prediction directories
path = kagglehub.dataset_download("puneet6060/intel-image-classification")
train_dir = os.path.join(path, os.path.join("seg_train", "seg_train"))
validation_dir = os.path.join(path, os.path.join("seg_test", "seg_test"))
pred_dir = os.path.join(path, os.path.join("seg_pred", "seg_pred"))

# Assigning an integer to each label; keys are the numbers, and values are the labels
label_dict = {}
for idx, item in enumerate(os.listdir(train_dir)):
    label_dict[idx] = item

if __name__=="__main__":
    # just a sample image of a building for testing [remove later]
    img = "C:/Users/satwi/.cache/kagglehub/datasets/puneet6060/intel-image-classification/versions/2/seg_train/seg_train/buildings/0.jpg"

    # Loading and specifying the intermediate model
    model = tf.keras.models.load_model("./convolution.keras")
    layer_name = 'dense'
    intermediate_layer_model = Model(inputs=model.inputs, outputs=model.get_layer(layer_name).output)

    # batch processing
    training_input, training_output = preprocessing(train_dir, label_dict)
    inter_output = intermediate_layer_model.predict(training_input)
    categories = np.digitize(inter_output, bins=np.linspace(0, 3, 7))
    ohe_input = np.array([np.eye(8)[items] for items in categories])
    ohe_input = ohe_input.reshape((ohe_input.shape[0], ohe_input.shape[1]*ohe_input.shape[2]))
    print(ohe_input.shape)
    print(training_output.shape)
    print(training_output[:2])
    print(ohe_input[:2])

    # weightless neural network architecture
    arch_i = [8 for i in range(256)]
    arch_z = [6]
    arch_c = []
    connector_function = "full_conn"
    description = "On top of CNN"
    connector_parameters = [256*6, 256*4, 256, 6]
    arch = ar.Arch(arch_i, arch_z, arch_c, connector_function, connector_parameters, description)

    agent = ao.Agent(arch, notes="On top of CNN", save_meta=False, _steps=50000)
    agent.full_conn_compress = True
    agent.next_state_batch(ohe_input, training_output, DD=False, unsequenced=True)

    agent.pickle()


