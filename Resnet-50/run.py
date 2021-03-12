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
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
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

test_data = create_data(TEST_PATH,0)

# Model

model = Sequential()
model.add(ResNet50(include_top = False, pooling = 'max', weights = 'imagenet'))
model.add(Dense(1, activation = 'sigmoid'))

model.layers[0].trainable = False
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

checkpointer = ModelCheckpoint(filepath = './dogcat_weights_best.hdf5', save_best_only = True, save_weights_only = True)

# Test & Output

model.load_weights("./dogcat_weights_best.hdf5")

test_df = pd.DataFrame({'file': os.listdir(TEST_PATH)})

test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

test_gen = test_datagen.flow_from_dataframe(test_df, directory = TEST_PATH, batch_size = BATCH_SIZE, x_col = 'file', y_col = None, class_mode = None, shuffle = False,
                                            img_size = (IMGSIZE, IMGSIZE))

prediction = model.predict_generator(test_gen)
prediction = prediction.clip(min = 0.005, max = 0.995)

submission_df = pd.read_csv('../sample_submission.csv')

for i, fname in enumerate(os.listdir(TEST_PATH)):
    index = int(fname[:fname.rfind('.')])
    submission_df.at[index-1, 'label'] = prediction[i]
submission_df.to_csv('Cats&DogsSubmission.csv', index=False)