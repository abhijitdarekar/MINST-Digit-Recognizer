import os
import sys

from src.logger import logging
from src.exception import CustomException
from src.config import CONFIG
from src.utils import MINSTModel

import torch
from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_store_path :str = CONFIG['MODEL_SAVE_PATH']
    learning_rate :float = CONFIG["LEARNING_RATE"]
    num_epochs :int = CONFIG['NUM_EPOCHS']

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def __call__(self, validation_loss):
        if self.min_validation_loss>validation_loss :
            logging.info("MODEL TRAINING : Loss changed.")
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif (self.min_validation_loss + self.min_delta)<=validation_loss:
            # print(f"Loss did not increase from {self.min_validation_loss:.5f} ")
            logging.info("MODEL TRAINING : No change in loss.")
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class Model:
    def __init__(self,train_loader,test_Loader):
        self.model_config = ModelConfig()

        self.train_loader= train_loader
        self.test_loader = test_Loader

        self.model = None
        self.criterion = None
        self.optimizer = None

        self.test_loss = []
        self.train_loss = []
        self.train_accuracy = []
        self.test_accuracy = []
        self.lr_change = []

    def __train_per_one_epoch(self):
        try:
            self.model.train()
            total_loss,total_accuracy = 0,0
            for data,label in self.train_loader:
                self.optimizer.zero_grad()
                out = self.model(data)
                loss = self.criterion(out, label)
                loss.backward()
                self.optimizer.step()
                total_loss+=loss.item()
                
                with torch.no_grad():
                    self.model.eval()
                    out = self.model(data)
                    _,top_class = out.topk(1, dim=1)
                    equals = top_class == label.view(*top_class.shape)
                    total_accuracy+=torch.mean(equals.type(torch.FloatTensor)).item()
                
                self.model.train()
            return total_accuracy, total_loss
        
        except Exception as e:
            logging.info(f"Caught Exception : {e}")
            raise CustomException(e,sys)
    
    def __test_per_one_epoch(self):
        try:
            total_accuracy, total_loss = 0,0
            with torch.no_grad():
                self.model.eval()
                for data, label in self.test_loader:
                    out = self.model(data)
                    _,top_class = out.topk(1, dim=1)
                    equals = top_class == label.view(*top_class.shape)
                    total_accuracy+=torch.mean(equals.type(torch.FloatTensor)).item()
                    total_loss+=self.criterion(out,label).item()

            return total_accuracy,total_loss
        except Exception as e:
            logging.info(f"Caught Exception : {e}")
            raise CustomException

    def initiate_traning(self,):

        try:
            self.model = MINSTModel()
            self.model = self.model()
            logging.info("MODEL TRAINING : Initated Model")

            size = 0
            for layer in self.model.parameters():
                size +=torch.numel(layer)
            logging.info(f"MODEL TRAINING : Total Number of parameters are {size}")
            
            self.criterion = torch.nn.CrossEntropyLoss()
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.model_config.learning_rate)
            logging.info("MODEL TRAINING : Initilized optimizer and loss functions")

            early_stopping = EarlyStopper(5,0)
            logging.info("MODEL TRAINING : Creating LR Scheduler.")
            scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.model_config.learning_rate, max_lr=0.1,step_size_up=5,mode="triangular2")

            logging.info("MODEL TRAINING : Initated Model Training")

            for epoch in range(self.model_config.num_epochs):

                train_acc, train_loss = self.__train_per_one_epoch()
                test_acc, test_Loss = self.__test_per_one_epoch()

                # Saving Metrics 
                self.train_accuracy.append(train_acc/len(self.train_loader))
                self.train_loss.append(train_loss/len(self.train_loader))
                self.test_accuracy.append(test_acc/len(self.test_loader))
                self.test_loss.append(test_Loss/len(self.test_loader))
                
                # Saving Learning Rate Changes
                self.lr_change.append(self.optimizer.param_groups[0]['lr'])
                
                # Stepping Scheduler
                scheduler.step()

                logging.info(f"MODEL TRAINING : Epoch:{epoch+1:3} | Training Loss {train_loss/len(self.train_loader):.5f}  | Test Loss {test_Loss/len(self.test_loader):.5f} | Train Accuracy {train_acc/len(self.train_loader):.5f} | Test Accuracy {test_acc/len(self.test_loader):.5f} | Learning Rate : {self.optimizer.param_groups[0]['lr']:.5f}")
                if early_stopping(test_Loss):
                    logging.info("MODEL TRAINING : Terminating Model training, as there is no change in Accuracy.....")
                    torch.save(self.model.state_dict(), self.model_config.model_store_path)
                    break

        except Exception as e:
            logging.info(f"Caught Exception : {e}")
            raise CustomException(e,sys)
            
