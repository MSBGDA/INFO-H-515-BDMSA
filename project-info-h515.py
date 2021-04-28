# importing necessary modules
import numpy as np
import pandas as pd
from PIL import Image
import os
import json

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from mpi4py import MPI  # MPI functions in Python

# initialising hyper paramenters
BATCH = 16
EPOCHS = 10

LR = 0.01
IM_SIZE = 32

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)  # checking if GPU is available if not use CPU

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size() # number workers

DATASET= "C:\\Users\\ivomb\\OneDrive\\Msc Data Science\\Second-Semester\\INFO-H-515-BigData-Distributed-Data-Management-and-Scalable-Analytics\\Practical_Sessions\\mini_herbarium\\train\\"  # training source
METADATA_FILE = DATASET + "metadata.json"

# Creating a custom dataset for image files
class GetData(Dataset):
    
    def __init__(self, Dir, FNames, Labels, Transform): # initialising class attributes
        self.dir = Dir
        self.fnames = FNames
        self.transform = Transform
        self.labels = Labels         
        
    def __len__(self): # returns number of samples
        return len(self.fnames)

    def __getitem__(self, index):        # retrives image sample
        x = Image.open(os.path.join(self.dir, self.fnames[index])) 
    
        if "train" in self.dir:             # transform
            return self.transform(x), self.labels[index]

# Node 0 loads the Herbarium dataset and gets the list of all the images in it
records_to_scatter = None


if rank == 0:
    import json                    # Module to read JSON files
    
    with open(METADATA_FILE) as f:
        d = json.load(f)           # Read metadata.json
        
    # importing image files and their annotations as pandas dataframe
    train_img = pd.DataFrame(d['images'])
    train_ann = pd.DataFrame(d['annotations']).drop(columns='image_id')
    train_df = train_img.merge(train_ann, on='id') # merging dataframes
          
    
    # Split them among <size> workers (node 0 included)
    elements_per_worker = train_df.shape[0] // size
    records_to_scatter = []

    for i in range(size):
        # fr and to: define a range of filenames to give to the i-th worker
        fr = i * elements_per_worker
        to = fr + elements_per_worker
        
        if i == size-1:
            # The last worker may have more images to process if <size> does not divide len(all_filenames)
            to = train_df.shape[0]
            
            records_to_scatter.append(train_df.iloc[fr:to,:])
        else:
            
            records_to_scatter.append(train_df.iloc[fr:to,:])
        
        

# Scatter the records
my_records = comm.scatter(records_to_scatter, root=0)


print('I am Node', rank, 'and I got', len(my_records), 'records to process')

# Every node reads all the images and averages them. The following piece of code is not surrounded by an "if", so it runs on all the nodes in parallel.
NUM_CL = len(my_records['category_id'].value_counts()) # requesting number of classes

X_Train, Y_Train =my_records['file_name'].values, my_records['category_id'].values # separating features and labels

# Getting dataset ready for training
Transform = transforms.Compose(
    [
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

trainset = GetData(DATASET, X_Train, Y_Train, Transform) # training dataset
trainloader = DataLoader(trainset, batch_size=BATCH, shuffle=True) # preparing data for taining

model = torchvision.models.resnet34() # instantiating the resnet34 neural network model
model.fc = nn.Linear(512, NUM_CL, bias=True) # applying linear transformation 
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss() # loss function
optimizer = torch.optim.Adam(model.parameters(), lr=LR) # Adam optimisation


