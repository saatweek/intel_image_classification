import ao_arch as ar
import ao_core as ao
import numpy as np
import tensorflow as tf
import pickle
import plotly as plt
import kagglehub
import os
from using_wnn import  preprocessing, transform_image, label_dict, validation_dir
from tensorflow.keras.models import Model

ml_model = tf.keras.models.load_model("./convolution.keras")

## For getting the accuracy of the wnn model
agent = ao.Agent.unpickle("./On top of CNN.ao.pickle")
layer_name = 'dense'
intermediate_layer_model = Model(inputs=ml_model.inputs, outputs=ml_model.get_layer(layer_name).output)

test_input, test_output = preprocessing(validation_dir, label_dict)
test_input = intermediate_layer_model.predict(test_input)
categories = np.digitize(test_input, bins=np.linspace(0, 3, 7))
ohe_input = np.array([np.eye(8)[items] for items in categories])
ohe_input = ohe_input.reshape((ohe_input.shape[0], ohe_input.shape[1] * ohe_input.shape[2]))
print(ohe_input.shape)
print(ohe_input[:2])
print(test_output.shape)
print(test_output[:2])
agent._update_neuron_data()
test_output_wnn = []


for idx in range(ohe_input.shape[0]):
    agent.reset_state()
    for s in range(3):
        print(f"{idx} of {ohe_input.shape[0]} done")
        res=agent.next_state(ohe_input[idx], DD=False)
    test_output_wnn.append(res)
test_output_wnn = np.asarray(test_output_wnn)
print(test_output_wnn.shape)
print(test_output_wnn[:2])
print(f"Accuracy of WNN model is : {(test_output*test_output_wnn).sum()/test_output.shape[0]}")


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
agent.pickle()