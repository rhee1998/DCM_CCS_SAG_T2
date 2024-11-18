import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle, sys, os, time, random, warnings, gc
from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.metrics import *

import tensorflow as tf
from tensorflow import data
from tensorflow import keras

from keras import backend as K
from keras.layers import CenterCrop, Lambda, RandomCrop, RandomFlip, RandomRotation, RandomZoom, RandomContrast, Resizing
from keras.layers import BatchNormalization, Concatenate, Dense, Dropout, GlobalAveragePooling2D
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam, SGD
from keras.metrics import AUC
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

from utils.dataset_tools import LoadDataset


# ============= #
# Library Setup #
# ============= #
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_ENABLE_ONE_DNN_OPTS'] = '0'
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

# Tensorflow Device Setup
physical_devices = tf.config.list_physical_devices('GPU')
try:tf.config.experimental.set_memory_growth(physical_devices[0], True)
except: pass

# Reproducibility
def seed_everything(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    tf.random.set_seed(random_seed)

# Tensorflow Device Setup
physical_devices = tf.config.list_physical_devices('GPU')
try:tf.config.experimental.set_memory_growth(physical_devices[0], True)
except: pass


# ======================== #
# File Paths & Hyperparams #
# ======================== #
TRIAL_DIR                   = sys.argv[1]
TRIAL_FOLDER                = TRIAL_DIR.split('/')[-1]
STUDY_DIR                   = TRIAL_DIR[:-(len(TRIAL_FOLDER) + 1)]

HPARAMS_ARCH_FILENAME       = f'{TRIAL_DIR}/hparams_arch.pkl'
HPARAMS_TRAIN_FILENAME      = f'{TRIAL_DIR}/hparams_train.pkl'
DATASET_CONFIG_FILENAME     = f'{TRIAL_DIR}/dataset_config.pkl'


# =========== #
# Build Model #
# =========== #
hparams_arch = pickle.load(open(HPARAMS_ARCH_FILENAME, 'rb'))

BASE_MODEL_FILENAME         = hparams_arch['BASE_MODEL_FILENAME']
ARCHITECTURE                = BASE_MODEL_FILENAME.split('/')[1]
FF_LAYER_LIST               = hparams_arch['FF_LAYER_LIST']
DROPOUT_RATE                = hparams_arch['DROPOUT_RATE']

# Load Base Model
if ARCHITECTURE == 'ResNet50':
	base_model = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet')
elif ARCHITECTURE == 'VGG16':
	base_model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')
elif ARCHITECTURE == 'InceptionV3':
	base_model = keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet')
elif ARCHITECTURE == 'Xception':
	base_model = keras.applications.Xception(include_top=False, weights='imagenet')
elif ARCHITECTURE == 'EfficientNetV2S':
	base_model = keras.applications.efficientnet_v2.EfficientNetV2S(include_top=False, weights='imagenet')
elif ARCHITECTURE == 'MobileNetV3Large':
	base_model = keras.applications.MobileNetV3Large(include_top=False, weights='imagenet')
elif ARCHITECTURE == 'ConvNeXtTiny':
	base_model = keras.applications.ConvNeXtTiny(include_top=False, weights='imagenet')
elif ARCHITECTURE == 'ConvNeXtSmall':
	base_model = keras.applications.ConvNeXtSmall(include_top=False, weights='imagenet')

for layer in base_model.layers:
    layer.trainable = False

# Add FF Layers
out = GlobalAveragePooling2D()(base_model.output)
for n_node in FF_LAYER_LIST:
    out = Dropout(rate=DROPOUT_RATE)(out)
    out = Dense(n_node, activation='relu')(out)
    out = BatchNormalization()(out)

# Final Output
out = Dropout(rate=DROPOUT_RATE)(out)
out = Dense(1)(out)

# Define Model
model = keras.Model(inputs=[base_model.input], outputs=[out])
model.save_weights(f'{TRIAL_DIR}/initial_weights.hdf5')


# ===================== #
# Train Hyperparameters #
# ===================== #
hparams_train = pickle.load(open(HPARAMS_TRAIN_FILENAME, 'rb'))

N_EPOCHS                    = hparams_train['N_EPOCHS']
BATCH_SIZE                  = hparams_train['BATCH_SIZE']
OPTIMIZER                   = hparams_train['OPTIMIZER']
LR_INIT                     = hparams_train['LR_INIT']
LR_REDUCE_PATIENCE          = hparams_train['LR_REDUCE_PATIENCE']
LR_REDUCE_FACTOR            = hparams_train['LR_REDUCE_FACTOR']


# ===================== #
# Dataset Configuration #
# ===================== #
dataset_config = pickle.load(open(DATASET_CONFIG_FILENAME, 'rb'))

DATASET_FILENAME            = dataset_config['DATASET_FILENAME']

N_FOLD                      = dataset_config['N_FOLD']
RANDOM_SEED                 = dataset_config['RANDOM_SEED']

INPUT_SHAPE                 = dataset_config['INPUT_SHAPE']
N_REPEAT                    = dataset_config['N_REPEAT']

ROTATION_LO                 = dataset_config['ROTATION_LO']
ROTATION_HI                 = dataset_config['ROTATION_HI']
ROTATION_FACTOR             = (ROTATION_LO / 360., ROTATION_HI / 360.)

ZOOM_LO                     = dataset_config['ZOOM_LO']
ZOOM_HI                     = dataset_config['ZOOM_HI']
ZOOM_FACTOR                 = (ZOOM_LO, ZOOM_HI)

BUFFER_SHUFFLE              = dataset_config['BUFFER_SHUFFLE']
BUFFER_PREFETCH             = dataset_config['BUFFER_PREFETCH']
AUTOTUNE                    = dataset_config['AUTOTUNE']


# Load Dataset
X_tval, y_tval, X_test, y_test = LoadDataset(DATASET_FILENAME)

# Data Augmentation
augment_train = tf.keras.Sequential([
    Lambda(lambda x: tf.cast(x, dtype=tf.float32)),
    RandomRotation(factor=ROTATION_FACTOR, fill_mode='reflect', interpolation='bilinear', seed=RANDOM_SEED),
    RandomCrop(height=int(INPUT_SHAPE[0] * 0.95), width=int(INPUT_SHAPE[1] * 0.95), seed=RANDOM_SEED),
    RandomZoom(height_factor=ZOOM_FACTOR, seed=RANDOM_SEED),
    Resizing(height=INPUT_SHAPE[0], width=INPUT_SHAPE[1])
])
augment_test = tf.keras.Sequential([
    Lambda(lambda x: tf.cast(x, dtype=tf.float32)),
    CenterCrop(height=int(INPUT_SHAPE[0] * 0.95), width=int(INPUT_SHAPE[1] * 0.95)),
    Resizing(height=INPUT_SHAPE[0], width=INPUT_SHAPE[1])
])

def PrepareDataset(ds, train=True):
    if train:
        ds = ds.repeat(N_REPEAT).shuffle(BUFFER_SHUFFLE)
        ds = ds.map(lambda X, y: (augment_train(X), y), num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(lambda X, y: (augment_test(X), y), num_parallel_calls=AUTOTUNE)

    return ds.batch(BATCH_SIZE).prefetch(buffer_size=BUFFER_PREFETCH)


# Prepare Test TFDS
ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
ds_test = PrepareDataset(ds_test, train=False)


# ======================= #
# Train Model (K-Fold CV) #
# ======================= #
sys.stdout.write(f'\n')
sys.stdout.write(f'======================\n')
sys.stdout.write(f'= STARTING NEW STUDY =\n')
sys.stdout.write(f'======================\n\n')

seed_everything(RANDOM_SEED)
kf = KFold(n_splits=N_FOLD, shuffle=True, random_state=RANDOM_SEED)
fold_auc_list, y_pred_proba_list = [], []

for i_fold, (train_idx, valid_idx) in enumerate(kf.split(X_tval)):
    sys.stdout.write(f'***************\n')
    sys.stdout.write(f'* Fold {i_fold + 1} of {N_FOLD} *\n')
    sys.stdout.write(f'***************\n\n')

    # Prepare TFDS
    X_train, y_train = X_tval[train_idx], y_tval[train_idx]
    X_valid, y_valid = X_tval[valid_idx], y_tval[valid_idx]

    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    ds_valid = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))

    ds_train = PrepareDataset(ds_train, train=True)
    ds_valid = PrepareDataset(ds_valid, train=False)


    # Load Initial Weights
    model.load_weights(f'{TRIAL_DIR}/initial_weights.hdf5')

    # Set Optimizer
    if OPTIMIZER == 'Adam': optimizer = Adam(learning_rate=LR_INIT)
    elif OPTIMIZER == 'SGD':  optimizer = SGD(learning_rate=LR_INIT)

    # Compile Model
    model.compile(
        loss        = BinaryCrossentropy(from_logits=True),
        optimizer   = optimizer,
        metrics     = [AUC(from_logits=True)]
    )

    # Set Callbacks
    weight_cache = f'{TRIAL_DIR}/weights_{i_fold}.hdf5'
    callback_model_ckpt     = ModelCheckpoint(monitor='val_loss', filepath=weight_cache, verbose=1, save_best_only=True)
    callback_early_stop     = EarlyStopping(monitor='val_loss', patience=LR_REDUCE_PATIENCE * 3, verbose=0, mode='auto')
    callback_reduce_lr      = ReduceLROnPlateau(monitor='val_loss', patience=LR_REDUCE_PATIENCE, factor=LR_REDUCE_FACTOR, verbose=1)

    # Train Model
    history = model.fit(
        ds_train,
        validation_data     = ds_valid,
        epochs              = N_EPOCHS,
        callbacks           = [
            callback_model_ckpt,
            callback_early_stop,
            callback_reduce_lr,
        ],
    )

    # Show Results (Fold)
    model.load_weights(weight_cache)
    y_pred_logit = model.predict(ds_test)
    y_pred_proba = 1. / (1. + np.exp(-y_pred_logit))
    y_pred_proba_list.append(y_pred_proba)

    fold_auc = roc_auc_score(y_test, y_pred_proba)
    fold_auc_list.append(fold_auc)
    sys.stdout.write(f'>>> FOLD [{i_fold + 1}/{N_FOLD}] TEST AUC = {fold_auc:.4f}\n\n')


# ================ #
# Ensemble Results #
# ================ #
y_pred_proba_list = np.array(y_pred_proba_list)
y_pred_proba = np.mean(y_pred_proba_list, axis=0)

test_auc = roc_auc_score(y_test, y_pred_proba)

# Print Results
print(f'*' * 100)
print(f'{TRIAL_DIR}\n')
print(f'TEST AUC')
for i in range(3):
    print(f'FOLD {i + 1}   : {fold_auc_list[i]:.4f}')

print(f'\nENSEMBLE : {test_auc:.4f}')
print(f'*' * 100)

# Rename Folder
TRIAL_FOLDER_NEW = f'auc={test_auc:.4f}_{TRIAL_FOLDER}'
TRIAL_DIR_NEW = f'{STUDY_DIR}/{TRIAL_FOLDER_NEW}'
os.rename(TRIAL_DIR, TRIAL_DIR_NEW)

# Run Garbage Collector
gc.collect()