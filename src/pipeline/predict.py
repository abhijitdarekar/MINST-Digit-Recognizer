
from src.logger import logging
from src.exception import CustomException
from src.config import CONFIG
from src.utils import MINSTModel,get_transforms

import torch
import sys
from PIL import Image
import numpy as np


class Predict:
    def __init__(self):
        self.model = MINSTModel()()
        _,self.transform = get_transforms()
        self.model.load_state_dict(torch.load(CONFIG['MODEL_SAVE_PATH']))
        self.model.eval()
    
    def __call__(self,image):
        try:
            image = image.convert('L')
            image = image.resize(CONFIG['IMAGE_SIZE'])

            image = self.transform(np.array(image))
            out = self.model(image.unsqueeze(0))
            _,top_class = out.topk(1, dim=1)
            return top_class.item()
        except Exception as e:
            logging.info(f"Caught Exception : {e}")
            return  CustomException(e,sys)

