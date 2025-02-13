import ao_arch as ar
import ao_core as ao
import numpy as np
import tensorflow as tf
import pickle
import plotly as plt
import kagglehub
import os
from using_wnn import  preprocessing, transform_image
from tensorflow.keras.models import Model


path = kagglehub.dataset_download("puneet6060/intel-image-classification")
validation_dir = os.path.join(path, os.path.join("seg_test", "seg_test"))


ml_model = tf.keras.models.load_model("./convolution.keras")

## For getting the accuracy of the wnn model
agent = ao.Agent.unpickle("./On top of CNN.ao.pickle")
layer_name = 'dense'
intermediate_layer_model = Model(inputs=ml_model.inputs, outputs=ml_model.get_layer(layer_name).output)

test_input, test_output = preprocessing(validation_dir)
test_input = intermediate_layer_model.predict(test_input)
categories = np.digitize(test_input, bins=np.linspace(0, 10, 128))
ohe_input =  np.array([np.eye(129)[items] for items in categories])
agent._update_neuron_data()
test_output_wnn = []


for idx in range(test_input.shape[0]):
    test_output_wnn.append(agent.next_state(ohe_input[idx]))
test_output_wnn = np.asarray(test_output_wnn)
print((test_output*test_output_wnn).sum()/test_output.shape[0])


## Accuracy for ML model
test_input = []
test_output = []
label_dict = {}
for idx, item in enumerate(os.listdir(validation_dir)):
    label_dict[idx] = item
for items in label_dict.keys():
    temp_path = os.path.join(validation_dir, label_dict[items])
    for pic in os.listdir(temp_path):
        test_input.append(transform_image(os.path.join(temp_path, pic)))
        test_output.append(items)

test_input = np.asarray(test_input)
test_output = np.asarray(test_output)
ml_model_output = ml_model.predict(test_input)
ml_out = []
for items in ml_model_output:
    ml_out.append(np.argmax(items))
ml_out = np.asarray(ml_out)
count=0
for idx in range(ml_out.shape[0]):
    if ml_out[idx]==test_output[idx]:
        count+=1
print(f'Accuracy of ML model is : {count/ml_out.shape[0]}')