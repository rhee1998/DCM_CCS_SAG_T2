import sys, os, time, random
from utils.sys_args import *

# Library Setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_ENABLE_ONE_DNN_OPTS'] = '0'


# Command Line Arguments
ARCHITECTURE = sys.argv[1]

# File Paths
PRETRAINED_MODEL_DIR      = f'pretrained/{ARCHITECTURE}'
PRETRAINED_MODEL_FILENAME = f'pretrained/{ARCHITECTURE}/base_model.keras'
if os.path.exists(PRETRAINED_MODEL_FILENAME):
	print(f'Pretrained [{ARCHITECTURE}] already exists.')
	sys.exit(0)


# (PRN) Load & Save Pretrained Model
print(f'Pretrained [{ARCHITECTURE}] does not exist. Continuing for download...')
os.makedirs(PRETRAINED_MODEL_DIR, exist_ok=True)

import tensorflow as tf
from tensorflow import keras

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

# Freeze Layers
for layer in base_model.layers:
	layer.trainable = False

# Save Model
base_model.save(PRETRAINED_MODEL_FILENAME)
