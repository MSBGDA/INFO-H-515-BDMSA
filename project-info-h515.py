# importing necessary modules
import numpy as np
from numpy.core.defchararray import mod
from numpy.lib.utils import source
import pandas as pd
from PIL import Image
import os
import json

import torch
import torch.nn as nn
from torch.nn import parameter
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from mpi4py import MPI  # MPI functions in Python

# initialising hyper paramenters
BATCH = 16
EPOCHS = 10

LR = 0.01
IM_SIZE = 32

DEVICE = torch.device("cpu")
torch.set_num_threads(1) #Disable parallelism in pytorch

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size() # number of workers

DATASET=  "C:\\Users\\ivomb\\OneDrive\\Msc Data Science\\Second-Semester\\INFO-H-515-BigData-Distributed-Data-Management-and-Scalable-Analytics\\Practical_Sessions\\mini_herbarium\\train\\"
#DATASET = "C:\\Users\\ivomb\\Downloads\\train\\"
METADATA_FILE = DATASET + "metadata.json"

# Creating a custom dataset for image files
#The GetData retrieves the dataset’s features and labels one sample at a time
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

# Node 0 loads the metadata and gets the list of all the images and their labels
records_to_scatter = None
NUM_CL = 0


if rank == 0:
    import json                  
    
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
    NUM_CL = len(train_df['category_id'].value_counts()) # requesting number of classes
        
        

# Scatter the records
my_records = comm.scatter(records_to_scatter, root=0)

#broadcast number of classes
NUM_CL = comm.bcast(NUM_CL, root=0)


print('I am Node', rank, 'and I got', len(my_records), 'records to process')




X_Train, Y_Train = my_records['file_name'].values, my_records['category_id'].values # separating features and labels

# Getting dataset ready for training
# perform some manipulation of the data and make it suitable for training.

Transform = transforms.Compose(
    [
    transforms.Resize((IM_SIZE, IM_SIZE)), # Resize the input image to the given size 
    transforms.ToTensor(), # ToTensor converts a PIL image or NumPy ndarray into a FloatTensor and scales the image’s pixel intensity values in the range [0., 1.]
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # Normalize a tensor image with mean and standard deviation. mean[1],...,mean[n]) and std: (std[1],..,std[n]) for n channels

trainset = GetData(DATASET, X_Train, Y_Train, Transform) # training dataset
trainloader = DataLoader(trainset, batch_size=BATCH, shuffle=True) # preparing data for taining
DEVICE = torch.device("cpu") # checking if GPU is available if not use CPU


#model = torchvision.models.resnet18(num_classes=NUM_CL) # instantiating the resnet34 neural network model
#model = model.to(DEVICE)

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CL)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net()

criterion = nn.CrossEntropyLoss() # loss function
optimizer = torch.optim.Adam(model.parameters(), lr=LR) # Adam optimisation

global_conv1 = []
global_conv2 = []
global_fc1 = []
global_fc2 = []
global_fc3 = []

for epoch in range(EPOCHS):
    tr_loss = 0.0

    model = model.train()

    for i, (images, labels) in enumerate(trainloader):        
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)       
        logits = model(images)       
        loss = criterion(logits, labels)
        optimizer.zero_grad()        
        loss.backward()

        global_conv1_w = comm.allgather(model.conv1.weight) # Receives  weights of conv1 from other nodes
        global_conv1_b = comm.allgather(model.conv1.bias) # Receives  bias of conv1 from other nodes
        mean_wgt  = torch.mean(torch.stack(global_conv1_w), dim=0) # Calculates a row mean for the weights
        mean_b  = torch.mean(torch.stack(global_conv1_b), dim=0) # Calculates the row mean for the bias

        for i in range(len(mean_wgt)):
            model.conv1.weight.detach()[i] = mean_wgt[i] # Updates local weight of conv1 layer with mean values
        for i in range(len(mean_b)):
            model.conv1.bias.detach()[i] = mean_b[i] # Updates local bias of conv1 layer with mean values

        # Same procedure for all the layers

        global_conv2_w = comm.allgather(model.conv2.weight)
        global_conv2_b = comm.allgather(model.conv2.bias)
        mean_wgt2  = torch.mean(torch.stack(global_conv2_w), dim=0)
        mean_b2 = torch.mean(torch.stack(global_conv2_b), dim=0)

        for i in range(len(mean_wgt2)):
            model.conv2.weight.detach()[i] = mean_wgt2[i]
        for i in range(len(mean_b2)):
            model.conv2.bias.detach()[i] = mean_b2[i]

        global_fc1_w = comm.allgather(model.fc1.weight)
        global_fc1_b = comm.allgather(model.fc1.bias)
        mean_wgtfc1  = torch.mean(torch.stack(global_fc1_w), dim=0)
        mean_bfc1 = torch.mean(torch.stack(global_fc1_b), dim=0)

        for i in range(len(mean_wgtfc1)):
            model.fc1.weight.detach()[i] = mean_wgtfc1[i]
        for i in range(len(mean_bfc1)):
            model.fc1.bias.detach()[i] = mean_bfc1[i]


        global_fc2_w = comm.allgather(model.fc2.weight)
        global_fc2_b = comm.allgather(model.fc2.bias)
        mean_wgtfc2  = torch.mean(torch.stack(global_fc2_w), dim=0)
        mean_bfc2 = torch.mean(torch.stack(global_fc2_b), dim=0)

        for i in range(len(mean_wgtfc2)):
            model.fc2.weight.detach()[i] = mean_wgtfc2[i]
        for i in range(len(mean_bfc2)):
            model.fc2.bias.detach()[i] = mean_bfc2[i]


        global_fc3_w = comm.allgather(model.fc3.weight)
        global_fc3_b = comm.allgather(model.fc3.bias)
        mean_wgtfc3  = torch.mean(torch.stack(global_fc3_w), dim=0)
        mean_bfc3 = torch.mean(torch.stack(global_fc3_b), dim=0)

        for i in range(len(mean_wgtfc3)):
            model.fc3.weight.detach()[i] = mean_wgtfc3[i]
        for i in range(len(mean_bfc3)):
            model.fc3.bias.detach()[i] = mean_bfc3[i]
        
       
        optimizer.step()     
        tr_loss += loss.detach().item()
       
    model.eval()
    print('Epoch: %d | Loss: %.4f'%(epoch, tr_loss ))

model_parameters =  comm.gather(model.state_dict(), root=0) 

       