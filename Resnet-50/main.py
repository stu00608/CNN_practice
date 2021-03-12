import numpy as np 
import pandas as pd
import cv2
from tqdm import tqdm
from random import shuffle
from zipfile import ZipFile
import glob
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras.applications import ResNet50                      # Pretrained Model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)

TRAIN_PATH = '../train'
TEST_PATH = '../test'
IMGSIZE = 224
BATCH_SIZE = 64
EPOCHS = 10

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

# Preprocessing

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

# Model

model = Sequential()
model.add(ResNet50(include_top = False, pooling = 'max', weights = 'imagenet'))
model.add(Dense(1, activation = 'sigmoid'))

model.layers[0].trainable = False
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

checkpointer = ModelCheckpoint(filepath = './dogcat_weights_best.hdf5', save_best_only = True, save_weights_only = True)

# Training

model.fit_generator(train_gen, epochs = EPOCHS, validation_data = valid_gen, callbacks = [checkpointer])

model.save('RESnetCNN.h5')

# Plot

loss = pd.DataFrame(model.history.history)
loss[['loss', 'val_loss']].plot()
loss[['acc', 'val_acc']].plot()