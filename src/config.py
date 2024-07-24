import os
import sys

CONFIG = {
    # Storage related config(s)
    "ARTIFACTS":'artifacts',

    "TRAIN_FILE_PATH":os.path.join("artifacts","train.csv"),
    "TEST_FILE_PATH":os.path.join("artifacts","test.csv"),

    "MODEL_NAME":os.path.join("artifacts",'model.pt'),
    
    # Modelling Related Config(s)
    "BATCH_SIZE":100,
    "USE_GPU":False,
    "TRAIN_SIZE":0.3,
    "IMAGE_SIZE":(28,28),

    "MODEL_SAVE_PATH":os.path.join("artifacts","pytorch_model.pt"),
    "LEARNING_RATE":0.0001,
    "NUM_EPOCHS":25
}