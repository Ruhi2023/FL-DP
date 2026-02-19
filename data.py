# File contains: Real brain tumor dataset loading with PyTorch integration for FL
# ** functions/classes
# BrainTumorDataset - implemented, untested, unbackedup
#   input: images(np.array), labels(np.array) | output: Dataset object
#   calls: torch.from_numpy | called by: get_client_dataloaders
#   process: wraps numpy arrays as PyTorch Dataset for DataLoader

# extract_dataset - implemented, untested, unbackedup
#   input: zip_path(str) | output: None
#   calls: zipfile.ZipFile.extractall | called by: load_data
#   process: extracts zip to data_dir if not already extracted

# load_data - implemented, untested, unbackedup
#   input: None | output: tuple (np.array images, np.array labels)
#   calls: extract_dataset, cv2.imread, cv2.resize, cv2.cvtColor | called by: split_data_for_clients
#   process: loads all images from Training and Testing subdirs, resizes to 224x224, converts BGR to RGB

# preprocess_data - implemented, untested, unbackedup
#   input: images(np.array) | output: np.array
#   calls: None | called by: split_data_for_clients
#   process: normalizes pixel values from [0,255] to [0,1] and converts to float32

# split_data_for_clients - implemented, untested, unbackedup
#   input: num_clients(int), zip_path(str) | output: None
#   calls: load_data, preprocess_data, train_test_split, np.save | called by: get_client_dataloaders
#   process: loads raw data, splits train/test, splits train across clients, saves to data_arrays/

# get_client_dataloaders - implemented, untested, unbackedup
#   input: num_clients(int), zip_path(str), batch_size(int) | output: list of DataLoader
#   calls: split_data_for_clients, np.load, BrainTumorDataset, DataLoader | called by: main.py
#   process: loads or creates client numpy arrays, wraps in Dataset, returns DataLoaders

import numpy as np
import os
import cv2
import zipfile
import shutil
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Constants
data_dir = "brain_tumor_data"
classes = ["glioma", "meningioma", "pituitary", "notumor"]
subdirs = ["Training", "Testing"]
save_dir = "data_arrays"


class BrainTumorDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.from_numpy(images).permute(0, 3, 1, 2).float()
        self.labels = torch.from_numpy(labels).long()
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def extract_dataset(zip_path):
    if os.path.exists(data_dir):
        print(f"Dataset directory '{data_dir}' already exists, skipping extraction.")
        return
    
    print(f"Extracting dataset from '{zip_path}' to '{data_dir}'...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extraction complete!")
    except zipfile.BadZipFile:
        raise ValueError(f"Error: '{zip_path}' is not a valid zip file.")
    except Exception as e:
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        raise RuntimeError(f"Error extracting zip file: {e}")


def load_data():
    images, labels = [], []
    
    for subdir in subdirs:
        subdir_path = os.path.join(data_dir, subdir)
        if not os.path.exists(subdir_path):
            print(f"Warning: Subdirectory {subdir_path} does not exist, skipping.")
            continue
        
        print(f"Loading data from {subdir} directory...")
        
        for label, category in enumerate(classes):
            class_path = os.path.join(subdir_path, category)
            
            if not os.path.exists(class_path):
                print(f"  Warning: Path {class_path} does not exist, skipping.")
                continue
            
            print(f"  Loading {category} images from {class_path}")
            
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if not os.path.isfile(img_path):
                    continue
                
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                img = cv2.resize(img, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(label)
    
    if len(images) == 0:
        raise ValueError("No images were loaded. Check dataset structure: expecting data_dir/Training|Testing/glioma|meningioma|pituitary|notumor/*.jpg")
    
    print(f"Successfully loaded {len(images)} images.")
    return np.array(images), np.array(labels)


def preprocess_data(images):
    return images.astype(np.float32) / 255.0


def split_data_for_clients(num_clients, zip_path):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    extract_dataset(zip_path)
    
    X, y = load_data()
    X = preprocess_data(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    np.save(os.path.join(save_dir, "X_test.npy"), X_test)
    np.save(os.path.join(save_dir, "y_test.npy"), y_test)
    print(f"Saved test data: {len(X_test)} samples")
    
    data_per_client = len(X_train) // num_clients
    
    for i in range(num_clients):
        start_idx = i * data_per_client
        end_idx = (i + 1) * data_per_client if i < num_clients - 1 else len(X_train)
        
        client_X = X_train[start_idx:end_idx]
        client_y = y_train[start_idx:end_idx]
        
        np.save(os.path.join(save_dir, f"client_{i}_X.npy"), client_X)
        np.save(os.path.join(save_dir, f"client_{i}_y.npy"), client_y)
        
        print(f"Saved client {i} data: {len(client_X)} samples")


def get_client_dataloaders(num_clients=3, zip_path="dataset.zip", batch_size=16):
    client_0_x = os.path.join(save_dir, "client_0_X.npy")
    
    if not os.path.exists(client_0_x):
        print("Client data not found. Preparing data from zip...")
        split_data_for_clients(num_clients, zip_path)
    
    dataloaders = []
    
    for i in range(num_clients):
        X = np.load(os.path.join(save_dir, f"client_{i}_X.npy"))
        y = np.load(os.path.join(save_dir, f"client_{i}_y.npy"))
        
        dataset = BrainTumorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        dataloaders.append(loader)
        
        print(f"Created DataLoader for client {i}: {len(dataset)} samples")
    
    return dataloaders