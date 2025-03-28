import os
import kagglehub
from PIL import Image
from sklearn.utils import shuffle
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import time

# Download data and set the training, validation and prediction directories
path = kagglehub.dataset_download("puneet6060/intel-image-classification")
train_dir = os.path.join(path, os.path.join("seg_train", "seg_train"))
validation_dir = os.path.join(path, os.path.join("seg_test", "seg_test"))
pred_dir = os.path.join(path, os.path.join("seg_pred", "seg_pred"))

# Assigning an integer to each label; keys are the numbers, and values are the labels
label_dict = {}
for idx, item in enumerate(os.listdir(train_dir)):
    label_dict[idx] = item


#Image Transformer
def transform_image(img_path:str) -> np.ndarray:
    """
    :param img_path: str : Path of the image
    :return: image in the form of a numpy array
    """
    img = Image.open(img_path).resize((150, 150))
    img_array = np.array(img) / 255.

    return img_array

#Image Preprocessing
def wnn_preprocessing(folder_path:str, model_path) -> tuple[np.ndarray, np.ndarray]:
    """
    :param folder_path:  str : takes the path of the folder in which all the images are present
    :param model_path: path of the model (str)
    :return:  All the images (in the form of a numpy array) and their corresponding one hot encoded labels (2 numpy arrays)
    """
    # Taking all the inputs and their labels
    input_arr = []
    label_arr = []
    start_time = time.time()
    print("Reading all the images...")
    for items in label_dict.keys():
        temp_path = os.path.join(folder_path, label_dict[items])
        for pic in os.listdir(temp_path):
            input_arr.append(transform_image(os.path.join(temp_path, pic)))
            label_arr.append(items)
    # Converting them into numpy arrays
    input_arr = np.asarray(input_arr)
    label_arr = np.asarray(label_arr)
    print(f"Took {time.time()-start_time:.2f} secs to read the {len(input_arr)} images")

    # Random shuffling
    print("Shuffling all the images...")
    input_arr, label_arr = shuffle(input_arr, label_arr)
    print("Images Shuffled!")


    # One Hot Encoding the label array
    print("Encoding all the labels...")
    label_col = one_hot_encode(label_arr)
    print("Encoding done!")

    # Converting the numpy arrays into tensors and normalizing them
    print("Preparing Images for CNN..")
    data = ImageDataset(input_arr, label_col)
    print("Loading Images in batches...")
    data_loader = DataLoader(data, batch_size=32, shuffle=False)
    model = CNN() # Initializing the model
    if torch.cuda.is_available():
        model.cuda()
    print("Loading Model...")
    model.load_state_dict(torch.load(model_path)) # Loading the trained weights and connections
    model.eval() # Putting the model in evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Getting CNN model output...")
    all_inter_outputs = []
    with torch.no_grad():
        for images, labels in data_loader:
            images= images.to(device)
            batch_inter_outputs, _ = model(images)
            all_inter_outputs.append(batch_inter_outputs.cpu().numpy())

    all_inter_outputs = np.concatenate(all_inter_outputs, axis=0)

    standardized_data = row_wise_standardize(all_inter_outputs)

    categories = np.digitize(standardized_data, bins=np.linspace(0, 3, 7))
    ohe_input = np.array([np.eye(8)[items] for items in categories])
    ohe_input = ohe_input.reshape((ohe_input.shape[0], ohe_input.shape[1] * ohe_input.shape[2]))
    # print(f"ohe_input.shape : {ohe_input.shape}")
    # print(f"label_col.shape : {label_col.shape}")
    # print(f"ohe_input[:5] : {ohe_input[:5]}")
    # print(f"label_col[:5] : {label_col[:5]}")
    return ohe_input, label_col
    # return all_inter_outputs, label_col
def cnn_preprocessing(folder_path):
    input_arr = []
    label_arr = []
    for items in label_dict.keys():
        temp_path = os.path.join(folder_path, label_dict[items])
        for pic in os.listdir(temp_path):
            input_arr.append(transform_image(os.path.join(temp_path, pic)))
            label_arr.append(items)

    input_arr = np.asarray(input_arr)
    label_arr = np.asarray(label_arr)

    input_arr, label_arr = shuffle(input_arr, label_arr)

    label_col = one_hot_encode(label_arr)

    return input_arr, label_col

def one_hot_encode(arr):
    encoded_array = np.zeros((arr.size, arr.max() + 1))
    encoded_array[np.arange(arr.size), arr] = 1

    return encoded_array

# Custom Dataset Class
class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2) # Convert (H, W, C) → (C, H, W)
        self.images = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(self.images)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Calculate the final feature map size
        final_size = 150 // 8  # (150 / 2 / 2 / 2) = 18
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * final_size * final_size, 256)  # 128 * 18 * 18
        )

        self.final_layers = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 6)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        intermediate_output = x
        x = self.final_layers(x)
        return intermediate_output, x

def row_wise_standardize(data):
   row_means = np.mean(data, axis=1, keepdims=True)
   row_stds = np.std(data, axis=1, keepdims=True)
   row_stds[row_stds == 0] = 1e-8  # Handle zero standard deviation
   standardized_data = (data - row_means) / row_stds
   return standardized_data