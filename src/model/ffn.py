import torch.nn as nn
import numpy as np


class LinearBLock(nn.Module):
    def __init__(self, in_features , out_features , dropout_rate):
        super().__init__()

        self.block =  nn.Sequential(
            nn.Linear(in_features , out_features, bias=True), #Linear Transformation (wx+b) -> xwT+b.
            nn.LayerNorm(out_features), # Normalization , Stabilizes activations (normalizes feature distribution).
            nn.Mish(), # Activation Func Smooth non Linearit for better gradient flow.
            nn.Dropout(p=dropout_rate) # Regularization , Randomly zeroes activations (a thats passed on to the next layer) during training.
        )

        nn.init.kaiming_normal_(self.block[0].weight , 
                                nonlinearity='relu'
                                ) 

    def forward(self,x):
        return self.block(x)

class IrisPyTorchFFN(nn.Module):

    def __init__(self, input_dim , hidden_dim , output_dim ,  dropout_rate):
        super(IrisPyTorchFFN, self).__init__()
        
        # Internal Feature Processing Pipeline using our custom blocks
        self.feature_extractor = nn.Sequential(
            LinearBLock(input_dim, hidden_dim ,dropout_rate),  #(B, 4) @ (4, 16) → (B, 16)  
            LinearBLock(hidden_dim, hidden_dim ,dropout_rate), #(B, 16) @ (16, 16) → (B, 16)
        )
    
        # Final Classification Head (No LayerNorm or Dropout here!)
        self.classification_head = nn.Linear(hidden_dim , output_dim) #(B, 16) @ (16, 3) → (B, 3)

        # Xavier Initialization for the final classification layer
        nn.init.xavier_normal_(self.classification_head.weight)
        

    def forward(self,x):
  
        features =  self.feature_extractor(x)#  Get features from the blocks
        logits = self.classification_head(features)#  Turn features into classification scores
        return logits
        










    
    def printshape(self):
        print(f"All The weights are already transposed {self.layer1.weight.shape},\nThe orignal shape was {self.layer1.weight.T.shape}")







#IrisPyTorchFFN(input_dim=4 ,hidden_dim=16 ,output_dim=3).printshape()