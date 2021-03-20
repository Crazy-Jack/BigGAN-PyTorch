import torch 
import torch.nn as nn 


class Eval:
    def __init__(self, layer_id):
        self.layer_id = layer_id
    


    def get_activation(self, ):
        """get activation for the target layer"""
        
