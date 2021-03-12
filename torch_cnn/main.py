import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
%matplotlib inline
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import time
import random
from tqdm import tqdm 

try:
    torch.cuda.is_available()
    print('\nCUDA is available!  Training on GPU ...')
except:
    print('\nCUDA is not available.  Training on CPU ...')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

img_files = os.listdir('../train/')
len(img_files)

img_files = list(filter(lambda x: x != 'train', img_files))
len(img_files)

# define draw
def plotCurve(x_vals, y_vals, 
                x_label, y_label, 
                x2_vals=None, y2_vals=None, 
                legend=None,
                figsize=(3.5, 2.5)):
    # set figsize
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.savefig(f'fig_{y_label}.png')
    if legend:
        plt.legend(legend)

def train_path(p): 
    return f"../train/{p}"

img_files = list(map(train_path, img_files))

class CatDogDataset(Dataset):
    def __init__(self, image_paths, transform):
        super().__init__()
        self.paths = image_paths
        self.len = len(self.paths)
        self.transform = transform
        
    def __len__(self): return self.len
    
    def __getitem__(self, index): 
        path = self.paths[index]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        label = 0 if 'cat' in path else 1
        return (image, label)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

random.shuffle(img_files)
train_files = img_files[:20000]
valid = img_files[20000:]

train_ds = CatDogDataset(train_files, transform)
train_dl = DataLoader(train_ds, batch_size=100)
len(train_ds), len(train_dl)

valid_ds = CatDogDataset(valid, transform)
valid_dl = DataLoader(valid_ds, batch_size=100)
len(valid_ds), len(valid_dl)

class CatAndDogNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size=(5, 5), stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=(5, 5), stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=(3, 3), padding=1)
        self.bn32 = nn.BatchNorm2d(32)
        self.bn64 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(in_features= 64 * 6 * 6, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=50)
        self.fc3 = nn.Linear(in_features=50, out_features=2)
        self.dp2 = nn.Dropout2d(0.2)
        self.dp5 = nn.Dropout2d(0.5)
        
        
    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2)
        
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2)

        X = F.relu(self.bn64(self.conv3(X)))
        X = F.max_pool2d(X, 2)
        
        X = X.view(X.shape[0], -1)
        X = F.relu(self.fc1(X))
        X = self.dp5(X)
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        
        # X = torch.sigmoid(X)
        return X

print("\n\nTraining...\n\n")

model = CatAndDogNet().cuda()
losses = []
val_losses = []
accuracies = []
val_accuracies = []
epoches = 5


loss_fn = nn.CrossEntropyLoss() # for multiple classification
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
for epoch in range(epoches):
    start = time.time()
    epoch_loss = 0
    epoch_accuracy = 0
    for X, y in tqdm(train_dl):
        X = X.cuda()
        y = y.cuda()
        preds = model(X)
        loss = loss_fn(preds, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        accuracy = ((preds.argmax(dim=1) == y).float().mean())
        epoch_accuracy += accuracy
        epoch_loss += loss
        # print('.', end='', flush=True)
        
    epoch_accuracy = epoch_accuracy/len(train_dl)
    accuracies.append(epoch_accuracy.item())
    epoch_loss = epoch_loss / len(train_dl)
    losses.append(epoch_loss.item())
    print("\nEpoch: {}\ntrain loss: {:.4f}, train accracy: {:.4f}\n".format(epoch+1, epoch_loss, epoch_accuracy),flush=True)

    with torch.no_grad():
        val_epoch_loss = 0
        val_epoch_accuracy = 0
        for val_X, val_y in tqdm(valid_dl):
            val_X = val_X.cuda()
            val_y = val_y.cuda()
            val_preds = model(val_X)
            val_loss = loss_fn(val_preds, val_y)

            val_epoch_loss += val_loss            
            val_accuracy = ((val_preds.argmax(dim=1) == val_y).float().mean())
            val_epoch_accuracy += val_accuracy
        val_epoch_accuracy = val_epoch_accuracy/len(valid_dl)
        val_accuracies.append(val_epoch_accuracy.item())
        val_epoch_loss = val_epoch_loss / len(valid_dl)
        val_losses.append(val_epoch_loss.item())
        print("\nvalid loss: {:.4f}, valid accracy: {:.4f}".format(val_epoch_loss, val_epoch_accuracy),flush=True)
    
    print("time: {:.4f}".format(time.time() - start),flush=True)

plotCurve(range(1, epoches + 1), losses,
              "epoch", "loss",
              range(1, epoches + 1), val_losses,
              ["train", "test"])

plotCurve(range(1, epoches + 1), accuracies,
              "epoch", "accuracy",
              range(1, epoches + 1), val_accuracies,
              ["train", "test"])



print("\n\nTesting...\n\n")

test_files = os.listdir('../test/')
len(test_files)

test_files = list(filter(lambda x: x != 'test', test_files))
len(test_files)

def test_path(p): 
    return f"../test/{p}"

test_files = list(map(test_path, test_files))

class TestCatDogDataset(Dataset):
    def __init__(self, image_paths, transform):
        super().__init__()
        self.paths = image_paths
        self.len = len(self.paths)
        self.transform = transform
        
    def __len__(self): return self.len
    
    def __getitem__(self, index): 
        path = self.paths[index]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        fileid = path.split('/')[-1].split('.')[0]
        return (image, fileid)

test_ds = TestCatDogDataset(test_files, transform)
test_dl = DataLoader(test_ds, batch_size=100)
len(test_ds), len(test_dl)

dog_probs = []
with torch.no_grad():
    for X, fileid in test_dl:
        X = X.cuda()
        preds = model(X)
        preds_list = F.softmax(preds, dim=1)[:, 1].tolist()
        dog_probs += list(zip(list(fileid), preds_list))
#         print(dog_probs)

dog_probs.sort(key = lambda d: int(d[0]))

ids = list(map(lambda x: x[0], dog_probs))
probs = list(map(lambda x: x[1], dog_probs))

output_df = pd.DataFrame({'id':ids,'label':probs})
output_df.to_csv('output1.csv', index=False)

output_df.head()