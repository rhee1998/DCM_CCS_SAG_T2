import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle, sys, os, time, random, warnings, gc, faulthandler
from tqdm import tqdm
from numba import cuda

from sklearn.model_selection import KFold
from sklearn.metrics import *

import tensorflow as tf
from tensorflow import keras

from keras.layers import CenterCrop, Lambda, RandomCrop, RandomFlip, RandomRotation, RandomZoom, RandomContrast, Resizing
from keras.layers import BatchNormalization, Concatenate, Dense, Dropout, GlobalAveragePooling2D
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam, SGD
from keras.metrics import AUC
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# ============= #
# Library Setup #
# ============= #
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_ENABLE_ONE_DNN_OPTS'] = '0'
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')
# faulthandler.enable()

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


# ===================== #
# Train Hyperparameters #
# ===================== #
hparams_train = pickle.load(open(HPARAMS_TRAIN_FILENAME, 'rb'))

N_EPOCHS                    = hparams_train['N_EPOCHS']
UNFREEZE_DEPTH              = hparams_train['UNFREEZE_DEPTH']
BATCH_SIZE                  = hparams_train['BATCH_SIZE']
OPTIMIZER                   = hparams_train['OPTIMIZER']
LR_INIT                     = hparams_train['LR_INIT']
LR_REDUCE_PATIENCE          = hparams_train['LR_REDUCE_PATIENCE']
LR_REDUCE_FACTOR            = hparams_train['LR_REDUCE_FACTOR']


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

# Unfreeze Layers

#for layer in base_model.layers[:-UNFREEZE_DEPTH]:
#    layer.trainable = False
#for layer in base_model.layers[-UNFREEZE_DEPTH:]:
#    layer.trainable = True
for layer in base_model.layers:
    layer.trainable = True

# Add FF Layers
out = GlobalAveragePooling2D()(base_model.output)
for n_node in FF_LAYER_LIST:
    out = Dropout(rate=DROPOUT_RATE, name=f'CustomDrop_{n_node}')(out)
    out = Dense(n_node, activation='relu', name=f'CustomFF_{n_node}')(out)
    out = BatchNormalization(name=f'CustomBN_{n_node}')(out)

# Final Output
out = Dropout(rate=DROPOUT_RATE, name='CustomDrop_final')(out)
out = Dense(1, name='CustomFF_final')(out)

# Define Model
model = keras.Model(inputs=[base_model.input], outputs=[out])

# Custom Callback Function (Implements prev_best)
class SaveModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, prev_best=None):
        super(SaveModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('SaveModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'
        
        if(prev_best == None):
            if mode == 'min':
                self.monitor_op = np.less
                self.best = np.Inf
            elif mode == 'max':
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                    self.monitor_op = np.greater
                    self.best = -np.Inf
                else:
                    self.monitor_op = np.less
                    self.best = np.Inf
        else:
            if mode == 'min':
                self.monitor_op = np.less
                self.best = prev_best
            elif mode == 'max':
                self.monitor_op = np.greater
                self.best = prev_best
            else:
                if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                    self.monitor_op = np.greater
                    self.best = prev_best
                else:
                    self.monitor_op = np.less
                    self.best = prev_best

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


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
with open(DATASET_FILENAME, 'rb') as f:
    ds = pickle.load(f)
    X_tval, y_tval = ds['X_tval'], ds['y_tval']
    X_test, y_test = ds['X_test'], ds['y_test']
    del ds

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

# Prepare TFDS for Test Set
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
    model.load_weights(f'{TRIAL_DIR}/weights_{i_fold}.hdf5')
    
    # Calculate prev_best Loss
    print('Calculating Previous Best Loss...')
    y_pred_logit_valid = model.predict(ds_valid)
    prev_best = keras.backend.binary_crossentropy(np.float32(y_valid.reshape((-1, 1))), y_pred_logit_valid, from_logits=True)
    prev_best = np.mean(prev_best).reshape((1,))[0]
    print(f'Previous Best Loss : {prev_best}\n')

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
    callback_model_ckpt     = SaveModelCheckpoint(monitor='val_loss', filepath=weight_cache, verbose=1, save_best_only=True, prev_best=prev_best)
    callback_early_stop     = EarlyStopping(monitor='val_loss', patience=LR_REDUCE_PATIENCE * 2, verbose=0, mode='auto')
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

    # Free Memory
    del X_train, y_train, X_valid, y_valid
    del ds_train, ds_valid

    # Show Results (Fold on Test)

    model.load_weights(weight_cache)
    y_pred_logit = model.predict(ds_test)
    y_pred_proba = 1. / (1. + np.exp(-y_pred_logit))
    y_pred_proba_list.append(y_pred_proba)

    fold_auc = roc_auc_score(y_test, y_pred_proba)
    fold_auc_list.append(fold_auc)
    sys.stdout.write(f'\n>>> FOLD [{i_fold + 1}/{N_FOLD}] TEST AUC = {fold_auc:.4f}\n\n')

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