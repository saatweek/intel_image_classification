import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from scipy.signal import convolve2d
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Conv2D
import plotly.express as px

def convolve(img : np.ndarray, filter : np.ndarray) -> np.ndarray:
    return convolve2d(img, filter, mode='same', boundary='fill', fillvalue=0)

filter_dict = {
    'sharpen' : np.array([[0, -1, 0],[-1, 5, -1], [0, -1, 0]]),
    'gaussian_blur' : np.array([[1/16, 2/16, 1/16], [2/16, 4/16, 2/16], [1/16, 2/16, 1/16]])
}

if __name__=="__main__":
    img = "C:/Users/satwi/.cache/kagglehub/datasets/puneet6060/intel-image-classification/versions/2/seg_train/seg_train/buildings/4.jpg"
    img = image.load_img(img, target_size=(150, 150, 3))
    img_array = image.img_to_array(img)
    img_array /= 255
    img_array = Conv2D.convolution_op(1, img_array, filter_dict['sharpen'])
    fig = px.imshow(img_array)
    fig.show()


