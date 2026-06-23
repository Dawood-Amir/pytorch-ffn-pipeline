import torch 

DEVICE =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
INPUT_DIM = 4
HIDDEN_DIM =16
OUTPUT_DIM =3 
DROPOUT_RATE = 0.15 # 15% of neurons will be randomly deactivated each batch

BATCH_SIZE = 32 
LEARNING_RATE = 0.01
WEIGHT_DECAY =0.01
EPOCHS = 150 
PATIENCE = 10

SEED = 42  # Reproducibility: controls random initialization, data shuffling, and train/validation/test splits.
MODEL_SAVE_PATH  = "best_pytorch_iris_model.pth"