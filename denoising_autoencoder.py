# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

import zipfile, os, cv2
from tqdm.auto import tqdm

import imgaug as ia
from imgaug import augmenters as iaa

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

sns.set_style('darkgrid')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

path_zip = '../input/denoising-dirty-documents/'
path = '/kaggle/working/'

with zipfile.ZipFile(path_zip + 'train.zip', 'r') as zip_ref:
    zip_ref.extractall(path)

with zipfile.ZipFile(path_zip + 'test.zip', 'r') as zip_ref:
    zip_ref.extractall(path)  
    
with zipfile.ZipFile(path_zip + 'train_cleaned.zip', 'r') as zip_ref:
    zip_ref.extractall(path)  
    
with zipfile.ZipFile(path_zip + 'sampleSubmission.csv.zip', 'r') as zip_ref:
    zip_ref.extractall(path)
    
train_img = sorted(os.listdir(path + '/train'))
train_cleaned_img = sorted(os.listdir(path + '/train_cleaned'))
test_img = sorted(os.listdir(path + '/test'))

imgs = [cv2.imread(path + 'train/' + f) for f in sorted(os.listdir(path + 'train/'))]
print('Median Dimensions:', np.median([len(img) for img in imgs]), np.median([len(img[0]) for img in imgs]))
del imgs

import matplotlib.pyplot as plt
imgs = [cv2.imread(path + 'train/' + f) for f in sorted(os.listdir(path + 'train/'))]
imgplot = plt.imshow(imgs[0])
del imgs

class DataLoader_Class:
    def __init__(self, path_zip , path):
        self.path_zip      = path_zip
        self.path          = path
        self.train_images  = []        
        self.train_cleaned = []
        self.test_images   = []
    def read_all_image(self):
        self.train_images_path  = [(self.path + 'train/' + f) for f in sorted(os.listdir(self.path + 'train/'))]
        self.train_cleaned_path = [(self.path + 'train_cleaned/' + f) for f in sorted(os.listdir(self.path + 'train_cleaned/'))]
        self.test_images_path   = [(self.path + 'test/' + f) for f in sorted(os.listdir(self.path + 'test/'))]
        
        transform = transforms.Compose([
            transforms.ToTensor()            
        ])
        
        for path in self.train_images_path:
            img = cv2.imread(path)
            img = np.asarray(img, dtype="uint8")
            img = cv2.resize(img, (540, 420) ,interpolation = cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transform(transforms.ToPILImage()(img))
                
            self.train_images.append(img)
        
        for path in self.train_cleaned_path:
            img = cv2.imread(path)
            img = np.asarray(img, dtype="uint8")
            img = cv2.resize(img, (540, 420) ,interpolation = cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transform(transforms.ToPILImage()(img))

            self.train_cleaned.append(img)
        
        for path in self.test_images_path:
            img = cv2.imread(path)
            img = np.asarray(img, dtype="uint8")
            img = cv2.resize(img, (540, 420) ,interpolation = cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transform(transforms.ToPILImage()(img))

            self.test_images.append(img)
        
        #convert data list to tensor 
        self.train_images = torch.stack(self.train_images)
        self.train_cleaned = torch.stack(self.train_cleaned)
        self.test_images = torch.stack(self.test_images)

        print(self.train_images.shape)
        print(self.train_cleaned.shape)
        print(self.test_images.shape)
        
        return self.train_images, self.train_cleaned, self.test_images 
        
    def see_an_image(self, number):
        f, axarr = plt.subplots(1,2, figsize=(50,100))
        axarr[0].imshow(self.train_images[number].permute(1,2,0))
        axarr[1].imshow(self.train_cleaned[number].permute(1,2,0))


data_loader = DataLoader_Class(path_zip, path)
train_set_x, train_set_y , test_set_x = data_loader.read_all_image()

from random import randint

index = set(randint(0, 144) for p in range(0, 32))
index = list(index)

train_test_x = train_set_x[index]
train_test_y = train_set_y[index]

train_test_x.shape

a = DataLoader_Class(path_zip, path)
b = a.read_all_image()
#a.see_an_image(5)

a.see_an_image(100)

"""
train_test_x train_test_y
train_set_x  train_set_y
test_set_x 
"""
batch_size = 2

from torch.utils.data import Dataset

class Dataset96(Dataset):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]
training_set = Dataset96(train_set_x, train_set_y)
train_loader = torch.utils.data.DataLoader(training_set,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_set = Dataset96(train_test_x, train_test_y)
test_loader  = torch.utils.data.DataLoader(test_set, 
                                           batch_size=batch_size, 
                                           shuffle=True)
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # encoder layers
        self.enc1 = nn.Conv2d(1, 512, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        
        # decoder layers
        self.dec1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  
        self.dec2 = nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2)
        self.dec3 = nn.ConvTranspose2d(128, 256, kernel_size=2, stride=2)
        self.dec4 = nn.ConvTranspose2d(256, 512, kernel_size=2, stride=2)
        self.out = nn.Conv2d(512, 1, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)        
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        # encode
        x = F.relu(self.enc1(x))
        x = self.bn1(x)
        x = self.pool(x)
        x = F.relu(self.enc2(x))
        x = self.bn2(x)
        x = self.pool(x)
        x = F.relu(self.enc3(x))
        x = self.bn3(x)
        x = self.pool(x)
        x = F.relu(self.enc4(x))
        x = self.bn4(x)
        x = self.pool(x) # the latent space representation
        
        # decode
        x = F.relu(self.dec1(x))
        x = (self.bn4(x))
        x = F.relu(self.dec2(x))
        x = (self.bn3(x))
        x = F.relu(self.dec3(x))
        x = (self.bn2(x))
        x = F.relu(self.dec4(x))
        x = (self.bn1(x))
        x = torch.sigmoid(self.out(x))
        return x
def train(model, num_epochs=5, batch_size=2, learning_rate=0.01):
    criterion = nn.MSELoss() # mean square error loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    
    outputs = []

    for epoch in range(num_epochs):
        for data in train_loader:
            img, _ = data
            
            #print(_)
            img, _ = img.cuda(), _.cuda()

            optimizer.zero_grad()
            recon = model(img)
            #print(recon.shape)
            loss = criterion(recon, _)
            loss.backward()
            optimizer.step()
            
            
        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
        outputs.append((epoch, img, recon),)
    return outputs

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

model = Autoencoder()
device = get_device()
print(device)
model.to(device)

outputs = train(model, num_epochs=20)

recon = outputs[1][2].detach().cpu().numpy()
for i in recon:
    print(i.shape)
    #i = i.reshape(-1,540)
    print(i)
    i =  torch.from_numpy(i)
    #lum_img = img[:, :, 0]
    print(type(i))
    plt.imshow(i.permute(1,2,0),cmap = "gray")
    break

