from mpi4py import MPI
from line_profiler import LineProfiler
import PIL
import PIL.Image
from PIL import Image
import os
import sys
import json
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# mpiexec -n 3 python -m mpi4py info-h515-pred.py
# mpiexec -n 3 kernprof -l -v  info-h515-pred.py 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
IM_SIZE = 32

Transform = transforms.Compose(
    [
    transforms.Resize((IM_SIZE, IM_SIZE)), # Resize the input image to the given size 
    transforms.ToTensor(), # ToTensor converts a PIL image or NumPy ndarray into a FloatTensor and scales the image’s pixel intensity values in the range [0., 1.]
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # Normalize a tensor image with mean and standard deviation. mean[1],...,mean[n]) and std: (std[1],..,std[n]) for n channels

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
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




#@profile
def node0(): 
    
    test_dir = "C:\\Users\\ivomb\\OneDrive\\Msc Data Science\\Second-Semester\\INFO-H-515-BigData-Distributed-Data-Management-and-Scalable-Analytics\\Practical_Sessions\\mini_herbarium\\train\\"
    # loading train images metadata
    with open(test_dir + 'metadata.json', "r", encoding="ISO-8859-1") as file:
        train = json.load(file)
    df_test = pd.DataFrame(train['images'])
    df_file = df_test.iloc[:,0] # extracting the file names
    
    
    for i in range(df_file.shape[0]):
        image_file = df_file[i]
        pilo_image = Image.open(os.path.join(test_dir, image_file))
        
        comm.send(pilo_image,dest=1) 
       
    print('Node 0: I sent a total of ',i+1, "files. How many did you receive node 1?")
    sys.stdout.flush() 

    # Tell Node 1 that we are done
    comm.send((None), dest=1)

#@profile
def node1():
    n = 0
    mini_batch = []
    while True:
        pilo_image = comm.recv(source=0)
        if pilo_image is None:
            break
        transform_image = Transform(pilo_image)
        mini_batch.append(transform_image.unsqueeze(0))#unsqueeze changes the shape of the tensor from 3 to 4
        if len(mini_batch) > 1000:
            
            comm.send(mini_batch[:len(mini_batch)//2],dest=2)  #sending first half to node 2
            comm.send(mini_batch[len(mini_batch)//2:],dest=3)  #sending last half to node 3
            n += len(mini_batch)
            mini_batch = [] 

    if len(mini_batch)>0: # flushing out any remaining images in the batche
        comm.send(mini_batch[:len(mini_batch)//2],dest=2)  #sending first half to node 2
        comm.send(mini_batch[len(mini_batch)//2:],dest=3)  #sending last half to node 3
        n += len(mini_batch)
        mini_batch = []

    # Tell Node 2 that we are done
    print('Node 1: I received a total of ',n,  " files and sent them to node 2 and 3")
    comm.send((None), dest=2)
    comm.send((None), dest=3)


#@profile
def node2():
    n = 0
    model_pred = torch.load('model.pth')
    while True:
        mini_batch = comm.recv(source=1)

        if mini_batch is None:
            break
        print("I'm node ", rank,"these are my predictions")
        for image in mini_batch:
            output = model_pred(image)
            _, predicted = torch.max(output, 1)
            n += 1 
            print(predicted)

#@profile
def node3():
    n = 0
    model_pred = torch.load('model.pth')
    while True:
        mini_batch = comm.recv(source=1)

        if mini_batch is None:
            break
        print("I'm node ", rank,"these are my predictions")
        for image in mini_batch:
            output = model_pred(image)
            _, predicted = torch.max(output, 1)
            n += 1 
            
            print(predicted)


if rank == 0:
    node0()
elif rank == 1:
    node1()
elif rank == 2:
    node2()
else:  node3()
