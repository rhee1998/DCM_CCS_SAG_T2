import numpy as np
import pandas as pd

import pickle, sys, os, time, random, warnings, gc
import h5py
from datetime import datetime
from tqdm import tqdm

# Reproducibility
SEED = 2026
def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

seed_everything()


# =============================================================================
# Base Architecture
PARENT_TRIAL                = 'fine_tune_2/ResNet50_241016_00/auc=0.8924'
# PARENT_TRIAL                = 'fine_tune_2/MobileNetV3Large_241016_02/auc=0.8901'
# PARENT_TRIAL                = 'fine_tune_0/VGG16_241015_00/auc=0.8565'
# PARENT_TRIAL                = 'fine_tune_2/EfficientNetV2S_241017_00/auc=0.9026'

N_TRIAL                     = 20

# Hyperparameter Search Space (FC Layers)
UNFREEZE_DEPTH_LO           = 30
UNFREEZE_DEPTH_HI           = 60

# Hyperparameter Search Space (Train)
N_EPOCHS                    = 100
BATCH_SIZE_LIST             = [4, 8, 16]
OPTIMIZER_LIST              = ['Adam', 'SGD']

LR_INIT_LO                  = 2e-5
LR_INIT_HI                  = 5e-4

LR_REDUCE_PATIENCE_LO       = 3
LR_REDUCE_PATIENCE_HI       = 5

LR_REDUCE_FACTOR_LO         = 0.10
LR_REDUCE_FACTOR_HI         = 0.20
# =============================================================================

# Locate Parent Trial Directory
parent_folder_prefix = PARENT_TRIAL.split('/')[-1]
parent_dir = ''

parent_study_dir = PARENT_TRIAL[:-(len(parent_folder_prefix) + 1)]
for folder in sorted(os.listdir(parent_study_dir), reverse=True):
    if folder.startswith(parent_folder_prefix):
        parent_dir = f'{parent_study_dir}/{folder}'
        break

if parent_dir == '':
    print('Parent trial does not exist...')
    sys.exit(0)


# Study Directory
finetune_cnt = int(PARENT_TRIAL.split('/')[0].split('_')[-1]) + 1
finetune_folder = f'fine_tune_{finetune_cnt}'
os.makedirs(finetune_folder, exist_ok=True)

study_arch = PARENT_TRIAL.split('/')[1].split('_')[0]
study_date = datetime.now().strftime('%y%m%d')
study_folder_prefix = f'{study_arch}_{study_date}'

study_num = 0
for folder in sorted(os.listdir(finetune_folder), reverse=True):
    if folder.startswith(study_folder_prefix):
        ch_idx = len(study_folder_prefix) + 1
        study_num = int(folder[ch_idx:ch_idx + 2]) + 1
        break

STUDY_DIR = f'{finetune_folder}/{study_folder_prefix}_{study_num:02d}'
os.makedirs(STUDY_DIR, exist_ok=True)


# Save Configuration (Search Space)
search_space_train = {
    'UNFREEZE_DEPTH_LO'     : UNFREEZE_DEPTH_LO,
    'UNFREEZE_DEPTH_HI'     : UNFREEZE_DEPTH_HI,

    'BATCH_SIZE_LIST'       : BATCH_SIZE_LIST,
    'OPTIMIZER_LIST'        : OPTIMIZER_LIST,

    'LR_INIT_LO'            : LR_INIT_LO,
    'LR_INIT_HI'            : LR_INIT_HI,

    'LR_REDUCE_PATIENCE_LO' : LR_REDUCE_PATIENCE_LO,
    'LR_REDUCE_PATIENCE_HI' : LR_REDUCE_PATIENCE_HI,
    
    'LR_REDUCE_FACTOR_LO'   : LR_REDUCE_FACTOR_LO,
    'LR_REDUCE_FACTOR_HI'   : LR_REDUCE_FACTOR_HI,
}

SEARCH_SPACE_TRAIN_FILENAME = f'{STUDY_DIR}/search_space_train.pkl'
pickle.dump(search_space_train, open(SEARCH_SPACE_TRAIN_FILENAME, 'wb'))


# Hyperparameter Random Search
def loguniform(low, high, size=None):
    return np.exp(np.random.uniform(np.log(low), np.log(high), size))

unfreeze_depth          = np.random.randint(low=UNFREEZE_DEPTH_LO, high=UNFREEZE_DEPTH_HI, size=N_TRIAL)
batch_size              = np.random.choice(BATCH_SIZE_LIST, N_TRIAL)
optimizer               = np.random.choice(OPTIMIZER_LIST, N_TRIAL)

lr_initial              = loguniform(low=LR_INIT_LO, high=LR_INIT_HI, size=N_TRIAL)
lr_reduce_patience      = np.random.randint(low=LR_REDUCE_PATIENCE_LO, high=LR_REDUCE_PATIENCE_HI + 1, size=N_TRIAL)
lr_reduce_factor        = np.random.uniform(low=LR_REDUCE_FACTOR_LO, high=LR_REDUCE_FACTOR_HI, size=N_TRIAL)


for i_trial in range(N_TRIAL):
    # Hyperparameters (Train)
    unfreeze_depth_     = unfreeze_depth[i_trial]
    batch_size_         = batch_size[i_trial]
    optimizer_          = optimizer[i_trial]
    lr_initial_         = lr_initial[i_trial]
    lr_reduce_patience_ = lr_reduce_patience[i_trial]
    lr_reduce_factor_   = lr_reduce_factor[i_trial]

    # Trial Folder
    trial_folder = ''
    #trial_folder += f'unfr={unfreeze_depth_}_'
    trial_folder += f'unfr=all_'
    trial_folder += f'batch={batch_size_}_optim={optimizer_}_'
    trial_folder += f'lr={lr_initial_:.5f}_pat={lr_reduce_patience_}_factor={lr_reduce_factor_:.3f}'
    trial_dir = f'{STUDY_DIR}/{trial_folder}'
    os.makedirs(trial_dir)

    # Copy Parent Trial Hyperparameters (FC Layers)
    PARENT_HPARAMS_ARCH_FILENAME = f'{parent_dir}/hparams_arch.pkl'
    TRIAL_HPARAMS_ARCH_FILENAME  = f'{trial_dir}/hparams_arch.pkl'
    os.system(f'cp {PARENT_HPARAMS_ARCH_FILENAME} {TRIAL_HPARAMS_ARCH_FILENAME}')
    
    # Copy Parent Trial Dataset Configurations
    PARENT_DATASET_CONFIG_FILENAME = f'{parent_dir}/dataset_config.pkl'
    TRIAL_DATASET_CONFIG_FILENAME  = f'{trial_dir}/dataset_config.pkl'
    os.system(f'cp {PARENT_DATASET_CONFIG_FILENAME} {TRIAL_DATASET_CONFIG_FILENAME}')

    # Wait for Ctrl+C
    print(f'Waiting for termination...') 
    time.sleep(1)
    print(f'Continuing trial...\n')

    # Copy Model Weights (Per Fold)
    N_FOLD = pickle.load(open(TRIAL_DATASET_CONFIG_FILENAME, 'rb'))['N_FOLD']
    for i in range(N_FOLD):
        copy_success = False
        while not copy_success:
            try:
                print(f'Copying model weights for [Fold {i + 1}]...')
                os.system(f'cp {parent_dir}/weights_{i}.hdf5 {trial_dir}/weights_{i}.hdf5')
                with h5py.File(f'{trial_dir}/weights_{i}.hdf5', 'r') as f:
                    print(list(f.keys()))
                copy_success = True
            
            except Exception as e:
                print(f'Error reading model file : {e}')
                print(f'Retrying file copy...\n')
        print(f'Copy Complete')
    

    # Save Train Hyperparameters
    hparams_train = {
        'PARENT_TRIAL_DIR'      : parent_dir,
        'N_EPOCHS'              : N_EPOCHS,
        'UNFREEZE_DEPTH'        : unfreeze_depth_,
        'BATCH_SIZE'            : batch_size_,
        'OPTIMIZER'             : optimizer_,

        'LR_INIT'               : lr_initial_,
        'LR_REDUCE_PATIENCE'    : lr_reduce_patience_,
        'LR_REDUCE_FACTOR'      : lr_reduce_factor_,
    }

    HPARAMS_TRAIN_FILENAME  = f'{trial_dir}/hparams_train.pkl'
    pickle.dump(hparams_train, open(HPARAMS_TRAIN_FILENAME, 'wb'))


    # Train Model
    os.system(f'python3 241015_FineTune_ModelTrain.py {trial_dir}')