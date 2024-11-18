import numpy as np
import pandas as pd

import pickle, sys, os, time, random, warnings, gc
from datetime import datetime
from tqdm import tqdm

# Reproducibility
SEED = 2024
def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

seed_everything()


# =============================================================================
# Base Architecture
ROOT_ARCHITECTURE           = 'ConvNeXtSmall'
N_TRIAL                     = 30

# Hyperparameter Search Space (FC Layers)
LAYER_CNT_LO                = 0
LAYER_CNT_HI                = 1

NODE_CNT_LO                 = 16
NODE_CNT_HI                 = 128

DROPOUT_RATE_LO             = 0.1
DROPOUT_RATE_HI             = 0.6


# Hyperparameter Search Space (Train)
N_EPOCHS                    = 100
BATCH_SIZE_LIST             = [16, 32, 64]
OPTIMIZER_LIST              = ['Adam', 'SGD']

LR_INIT_LO                  = 1e-3
LR_INIT_HI                  = 5e-2

LR_REDUCE_PATIENCE_LO       = 3
LR_REDUCE_PATIENCE_HI       = 6

LR_REDUCE_FACTOR_LO         = 0.05
LR_REDUCE_FACTOR_HI         = 0.20

# Dataset Configuration
DATASET_FILENAME            = 'dataset/ALL/dataset.pkl'

N_FOLD                      = 3
RANDOM_SEED                 = 98

INPUT_SHAPE                 = (224, 112)
N_REPEAT                    = 3

ROTATION_LO                 = -10
ROTATION_HI                 = +10

ZOOM_LO                     = -0.1
ZOOM_HI                     = +0.1

BUFFER_SHUFFLE              = 1000
BUFFER_PREFETCH             = 3
AUTOTUNE                    = 12
# =============================================================================


# Load Pretrained Model
os.system(f'python3 241013_LoadPretrainedModel.py {ROOT_ARCHITECTURE}')


# Study Directory
study_date = datetime.now().strftime('%y%m%d')
study_folder_prefix = f'{ROOT_ARCHITECTURE}_{study_date}'

study_num = 0
for folder in sorted(os.listdir('fine_tune_0'), reverse=True):
    if folder.startswith(study_folder_prefix):
        ch_idx = len(study_folder_prefix) + 1
        study_num = int(folder[ch_idx:ch_idx + 2]) + 1
        break

STUDY_DIR = f'fine_tune_0/{study_folder_prefix}_{study_num:02d}'
os.makedirs(STUDY_DIR, exist_ok=True)


# Save Configuration (Search Space)
search_space_arch = {
    'LAYER_CNT_LO'          : LAYER_CNT_LO,
    'LAYER_CNT_HI'          : LAYER_CNT_HI,
    
    'NODE_CNT_LO'           : NODE_CNT_LO,
    'NODE_CNT_HI'           : NODE_CNT_HI,

    'DROPOUT_RATE_LO'       : DROPOUT_RATE_LO,
    'DROPOUT_RATE_HI'       : DROPOUT_RATE_HI,
}
search_space_train = {
    'BATCH_SIZE_LIST'       : BATCH_SIZE_LIST,
    'OPTIMIZER_LIST'        : OPTIMIZER_LIST,

    'LR_INIT_LO'            : LR_INIT_LO,
    'LR_INIT_HI'            : LR_INIT_HI,

    'LR_REDUCE_PATIENCE_LO' : LR_REDUCE_PATIENCE_LO,
    'LR_REDUCE_PATIENCE_HI' : LR_REDUCE_PATIENCE_HI,
    
    'LR_REDUCE_FACTOR_LO'   : LR_REDUCE_FACTOR_LO,
    'LR_REDUCE_FACTOR_HI'   : LR_REDUCE_FACTOR_HI,
}

SEARCH_SPACE_ARCH_FILENAME  = f'{STUDY_DIR}/search_space_arch.pkl'
SEARCH_SPACE_TRAIN_FILENAME = f'{STUDY_DIR}/search_space_train.pkl'

pickle.dump(search_space_arch, open(SEARCH_SPACE_ARCH_FILENAME, 'wb'))
pickle.dump(search_space_train, open(SEARCH_SPACE_TRAIN_FILENAME, 'wb'))


# Hyperparameter Random Search
def loguniform(low, high, size=None):
    return np.exp(np.random.uniform(np.log(low), np.log(high), size))

batch_size              = np.random.choice(BATCH_SIZE_LIST, N_TRIAL)
optimizer               = np.random.choice(OPTIMIZER_LIST, N_TRIAL)

lr_initial              = loguniform(low=LR_INIT_LO, high=LR_INIT_HI, size=N_TRIAL)
lr_reduce_patience      = np.random.randint(low=LR_REDUCE_PATIENCE_LO, high=LR_REDUCE_PATIENCE_HI + 1, size=N_TRIAL)
lr_reduce_factor        = np.random.uniform(low=LR_REDUCE_FACTOR_LO, high=LR_REDUCE_FACTOR_HI, size=N_TRIAL)

ff_layer_cnt            = np.random.randint(low=LAYER_CNT_LO, high=LAYER_CNT_HI + 1, size=N_TRIAL)
dropout_rate            = np.random.uniform(low=DROPOUT_RATE_LO, high=DROPOUT_RATE_HI, size=N_TRIAL)


for i_trial in range(N_TRIAL):
    # Hyperparameters (FC Layers)
    ff_layer_cnt_       = ff_layer_cnt[i_trial]
    ff_layer_list_      = sorted(np.random.randint(low=NODE_CNT_LO, high=NODE_CNT_HI + 1, size=ff_layer_cnt_), reverse=True)
    dropout_rate_       = dropout_rate[i_trial]

    # Hyperparameters (Train)
    batch_size_         = batch_size[i_trial]
    optimizer_          = optimizer[i_trial]
    lr_initial_         = lr_initial[i_trial]
    lr_reduce_patience_ = lr_reduce_patience[i_trial]
    lr_reduce_factor_   = lr_reduce_factor[i_trial]

    # Trial Folder
    trial_folder = ''
    trial_folder += f'batch={batch_size_}_optim={optimizer_}_'
    trial_folder += f'lr={lr_initial_:.5f}_pat={lr_reduce_patience_}_factor={lr_reduce_factor_:.3f}_'
    trial_folder += f'ff={ff_layer_list_}_drop={dropout_rate_:.3f}'
    trial_dir = f'{STUDY_DIR}/{trial_folder}'
    os.makedirs(trial_dir)

    # Save Train Configurations
    hparams_arch = {
        'BASE_MODEL_FILENAME'   : f'pretrained/{ROOT_ARCHITECTURE}/base_model.keras',
        'FF_LAYER_LIST'         : ff_layer_list_,
        'DROPOUT_RATE'          : dropout_rate_,
    }
    hparams_train = {
        'N_EPOCHS'              : N_EPOCHS,
        'BATCH_SIZE'            : batch_size_,
        'OPTIMIZER'             : optimizer_,

        'LR_INIT'               : lr_initial_,
        'LR_REDUCE_PATIENCE'    : lr_reduce_patience_,
        'LR_REDUCE_FACTOR'      : lr_reduce_factor_,
    }
    dataset_config = {
        'DATASET_FILENAME'      : DATASET_FILENAME,

        'N_FOLD'                : N_FOLD,
        'RANDOM_SEED'           : RANDOM_SEED,
        
        'INPUT_SHAPE'           : INPUT_SHAPE,
        'N_REPEAT'              : N_REPEAT,
        
        'ROTATION_LO'           : ROTATION_LO,
        'ROTATION_HI'           : ROTATION_HI,

        'ZOOM_LO'               : ZOOM_LO,
        'ZOOM_HI'               : ZOOM_HI,

        'BUFFER_SHUFFLE'        : BUFFER_SHUFFLE,
        'BUFFER_PREFETCH'       : BUFFER_PREFETCH,
        'AUTOTUNE'              : AUTOTUNE
    }

    HPARAMS_ARCH_FILENAME   = f'{trial_dir}/hparams_arch.pkl'
    HPARAMS_TRAIN_FILENAME  = f'{trial_dir}/hparams_train.pkl'
    DATASET_CONFIG_FILENAME = f'{trial_dir}/dataset_config.pkl'

    pickle.dump(hparams_arch, open(HPARAMS_ARCH_FILENAME, 'wb'))
    pickle.dump(hparams_train, open(HPARAMS_TRAIN_FILENAME, 'wb'))
    pickle.dump(dataset_config, open(DATASET_CONFIG_FILENAME, 'wb'))

    # Train Model
    os.system(f'python3 241013_Root_ModelTrain.py {trial_dir}')