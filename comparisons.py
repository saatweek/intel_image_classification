import ao_arch as ar
import ao_core as ao
import numpy as np
import tensorflow as tf
import pickle
import plotly as plt
import kagglehub
import os
from using_wnn import  preprocessing
from tensorflow.keras.models import Model

path = kagglehub.dataset_download("puneet6060/intel-image-classification")
validation_dir = os.path.join(path, os.path.join("seg_test", "seg_test"))


ml_model = tf.keras.models.load_model("./convolution.keras")
agent = ao.Agent.unpickle("./On top of CNN.ao.pickle")
layer_name = 'dense'
intermediate_layer_model = Model(inputs=ml_model.inputs, outputs=ml_model.get_layer(layer_name).output)

test_input, test_output = preprocessing(validation_dir)
test_input = intermediate_layer_model.predict(test_input)
agent._update_neuron_data()
test_output_wnn = []
for idx in range(test_input.shape[0]):
    test_output_wnn.append(agent.next_state(test_input[idx]))
test_output_wnn = np.asarray(test_output_wnn)
print((test_output*test_output_wnn).sum()/test_output.shape[0])