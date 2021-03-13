# Convolutional-Neural-Network Practice

> Source : Kaggle [Dogs vs. Cats Redux : Kernels Edition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/)
> 
> OS Platform : Ubuntu 20.04 LTS
> PyTorch version : 1.14
> Tensorflow version : 1.14.0
> Python version: 3.6.13
> CUDA/cuDNN version: CUDA 10.0 / cuDNN 7.6.0
> GPU model and memory: GeForce GTX 1060 6GB

:::success
檔案已經事先下載並解壓好
:::

## Basic CNNs

### Preprocessing

* 首先把`train`,`test`兩個資料夾內的檔案讀入程式(Full Color),訓練資料需要標記是貓或是狗.
* 將圖片重新調整大小到設定好的`IMGSIZE`,並對圖片標準化.
* 將訓練資料分為訓練集和驗證集,這裡使用0.25的驗證資料比例.

```python=
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

```

### Model

![](https://i.imgur.com/GbMIOiF.png)

* 卷積層負責提取圖片的部位特徵,再透過ReLu函數讓物體的形狀明顯化,提高準確度.
* 池化層負責挑選固定區域的最大值(特徵),組合成新的類似放大過的圖片.
* 經過反覆2～3次,在第三層的ReLi層之前做一次BatchNormalization,目的是控制較大的Batch Size,提昇準確率.
* 最後的結果攤平後送進基本Linear的網路當中訓練,此時應該會抓到夠多的特徵讓model學習.
* 在FC的過程中做了一次Dropout,提昇validation/test case的準確率.
* 注意各層連接的input和output數要對到.

```python=

class CatAndDogNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size=(5, 5), stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=(5, 5), stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=(3, 3), padding=1)
        self.bn64 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(in_features= 64 * 6 * 6, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=50)
        self.fc3 = nn.Linear(in_features=50, out_features=2)
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
        
        return X
```

### Train

```python=
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
```

### Result

![](https://i.imgur.com/KmMCoqr.png)

## ResNet50 Pretrained Model

![](https://i.imgur.com/T1EtNxS.png)


> ResNet : 殘差神經網絡
> 在layer太多的情況下,神經網路太多層容易導致退化(並非overfitting...等等),殘差指的就是當前的特徵`H(x)`減去原本的輸入`x`得出`F(x)=H(x)-x`,所以在神經網路forward的狀況下`x=F(x)+x`.
> 
> ![](https://i.imgur.com/qDf6CiI.png)
> 
> 如果過了一組layer但沒有新的特徵變化,至少出來的結果會維持不變,提高學習的準確率.

### Preprocessing

* 

```python=
is_dog = lambda category : int(category=='dog')

def create_data(path,is_Train=True):
    data = [] # 2-D for pandas structure
    img_list = os.listdir(path)
    for name in tqdm(img_list):
        img_addr = os.path.join(path,name)
        img = cv2.imread(img_addr,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMGSIZE, IMGSIZE) )
        if( is_Train ):
            label = is_dog(name.split(".")[0])
        else:
            label = name.split(".")[0]
        data.append([ np.array(img), np.array(label) ])
    
    shuffle(data)
    return data
    
train_data = create_data(TRAIN_PATH)

df = pd.DataFrame( {'file':os.listdir(TRAIN_PATH), 'label':[ str(is_dog(img.split(".")[0])) for img in os.listdir(TRAIN_PATH)]} )
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'],random_state=123)

print(train_df.shape)
print(val_df.shape)


#We'll perform individually on train and validation set.
train_datagen = ImageDataGenerator(rotation_range = 10, zoom_range = 0.1, horizontal_flip = True, fill_mode = 'nearest', 
                                   width_shift_range = 0.1, height_shift_range = 0.1, preprocessing_function = preprocess_input)

#flow_from_dataframe() method will accept dataframe with filenames as x_column and labels as y_column to generate mini-batches
train_gen = train_datagen.flow_from_dataframe(train_df, directory = TRAIN_PATH, x_col = 'file', y_col = 'label', target_size = (IMGSIZE,IMGSIZE),
                                              batch_size = BATCH_SIZE, class_mode='binary')

#we do not augment validation data.
valid_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

valid_gen = valid_datagen.flow_from_dataframe(val_df, directory = TRAIN_PATH, x_col = 'file', y_col = 'label', target_size = (IMGSIZE,IMGSIZE),
                                              batch_size = BATCH_SIZE, class_mode='binary')

```

### Model

* 

```python=
model = Sequential()
model.add(ResNet50(include_top = False, pooling = 'max', weights = 'imagenet'))
model.add(Dense(1, activation = 'sigmoid'))

model.layers[0].trainable = False
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

checkpointer = ModelCheckpoint(filepath = './dogcat_weights_best.hdf5', save_best_only = True, save_weights_only = True)

```

### Train

* 

```python=
model.fit_generator(train_gen, epochs = EPOCHS, validation_data = valid_gen, callbacks = [checkpointer])

model.save('RESnetCNN.h5')
```

### Result

![](https://i.imgur.com/MuUzrnF.png)

![](https://i.imgur.com/tESTcdT.png)

## Kaggle Score

![](https://i.imgur.com/7V3UJb9.png)

## Resource

* [Dogs vs Cats PyTorch CNN without transfer learning](https://www.kaggle.com/chriszou/dogs-vs-cats-pytorch-cnn-without-transfer-learning)
* [[資料分析&機器學習] 第5.1講: 卷積神經網絡介紹(Convolutional Neural Network)](https://medium.com/jameslearningnote/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC5-1%E8%AC%9B-%E5%8D%B7%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E7%B5%A1%E4%BB%8B%E7%B4%B9-convolutional-neural-network-4f8249d65d4f)
* [[ML筆記] Batch Normalization
](https://violin-tao.blogspot.com/2018/02/ml-batch-normalization.html)
* [ML Lecture 10: Convolutional Neural Network (Hung-yi Lee)](https://www.youtube.com/watch?v=FrKWiRv254g)
* [Cat or Dog - Transfer Learning using ResNets](https://www.kaggle.com/sanchitvj/cat-or-dog-transfer-learning-using-resnets?select=resnet50_weights_tf_dim_ordering_tf_kernels.h5)
* [你必须要知道CNN模型：ResNet](https://zhuanlan.zhihu.com/p/31852747)

###### tags: `PyTorch` `python` `MachineLearning` `CNN` `ResNet50` `Tensorflow`