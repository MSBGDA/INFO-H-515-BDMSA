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

# initialising hyper paramenters
BATCH = 2 #the number of data samples propagated through the network before the parameters are updated
EPOCHS = 10 #the number times to iterate over the dataset

LR = 0.01
IM_SIZE = 32

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # checking if GPU is available if not use CPU
torch.set_num_threads(1) # number of threads to one
TRAIN_DIR = "C:\\Users\\ivomb\\OneDrive\\Msc Data Science\\Second-Semester\\INFO-H-515-BigData-Distributed-Data-Management-and-Scalable-Analytics\\Practical_Sessions\\mini_herbarium\\train\\"    # training source

# loading train images metadata
with open(TRAIN_DIR + 'metadata.json', "r", encoding="ISO-8859-1") as file:
    train = json.load(file) 

# importing image file name and their annotations as pandas dataframe
train_img = pd.DataFrame(train['images'])
train_ann = pd.DataFrame(train['annotations']).drop(columns='image_id')
train_df = train_img.merge(train_ann, on='id') # merging dataframes

print('\nNumber of image files', len(train_df))

NUM_CL = len(train_df['category_id'].value_counts()) # requesting number of classes

X_Train, Y_Train = train_df['file_name'].values, train_df['category_id'].values # separating features and labels

Transform = transforms.Compose(
    [
    transforms.Resize((IM_SIZE, IM_SIZE)), # Resize the input image to the given size 
    transforms.ToTensor(), # ToTensor converts a PIL image or NumPy ndarray into a FloatTensor and scales the image’s pixel intensity values in the range [0., 1.]
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # Normalize a tensor image with mean and standard deviation. mean[1],...,mean[n]) and std: (std[1],..,std[n]) for n channels

# Creating a custom dataset for image files
#The Dataset retrieves our dataset’s features and labels one sample at a time
class GetData(Dataset):
    
    def __init__(self, Dir, FNames, Labels, Transform): # initialising class attributes
        self.dir = Dir
        self.fnames = FNames
        self.transform = Transform
        self.labels = Labels         
        
    def __len__(self): # returns number of samples
        return len(self.fnames)

    def __getitem__(self, index):        # retrives image sample
        x = Image.open(os.path.join(self.dir, self.fnames[index])) #opening images with pilow
        # transform
        return self.transform(x), self.labels[index]

trainset = GetData(TRAIN_DIR, X_Train, Y_Train, Transform) # training dataset
trainloader = DataLoader(trainset, batch_size=BATCH, shuffle=True) # preparing data for taining

#Model 1
model = torchvision.models.vgg11(num_classes=NUM_CL) # instantiating the resnet34 neural network model
model = model.to(DEVICE)

import torch.nn as nn
import torch.nn.functional as F

#model 2
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

# Choose 1 optimiser: hold the current model state and will update the parameters based on the computed gradients.
optimizer = torch.optim.Adam(model.parameters(), lr=LR) # Adam optimisation
#optimizer = torch.optim.Adagrad(model.parameters(), lr=LR)

print("\ntraining Process")
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
        optimizer.step()

        tr_loss += loss.detach().item()
    
    model.eval()
    print('Epoch: %d | Loss: %.4f'%(epoch, tr_loss ))

    torch.save(model, 'model.pth')

#Prediction

print("\nPredictions")

# Loading the saved model
model_1 = torch.load('model.pth')

#Sample image for prediction
x = Transform(Image.open("C:\\Users\\ivomb\\OneDrive\\Msc Data Science\\Second-Semester\\INFO-H-515-BigData-Distributed-Data-Management-and-Scalable-Analytics\\Practical_Sessions\\mini_herbarium\\train\\images\\000\\01\\230965.jpg"))

x= x.unsqueeze(0) 

#current class
print("\nImage class", train_df[train_df['file_name'] == "images/000/01/230965.jpg"])

# prediction
output = model_1(x)

# Requesting the class with the highest prediction probability
_, predicted = torch.max(output, 1)

#predicted class
print("\n predicted class",predicted)


# Image Two
x2 =  Transform(Image.open("C:\\Users\\ivomb\\OneDrive\\Msc Data Science\\Second-Semester\\INFO-H-515-BigData-Distributed-Data-Management-and-Scalable-Analytics\\Practical_Sessions\\mini_herbarium\\train\\images\\000\\00\\1360648.jpg"))
x2= x2.unsqueeze(0) 
        
#current class
print("\nImage class", train_df[train_df['file_name'] == "images/000/00/1360648.jpg"])

# prediction
output = model_1(x2)

# Requesting the class with the highest prediction probability
_, predicted = torch.max(output, 1)

#predicted class
print("\n predicted class",predicted)