import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim

import config as config 
from data.data_loader import get_data_loaders
from model.ffn import IrisPyTorchFFN

from utils.seed_utils import set_seed



def train_one_run(seed):
    set_seed(seed)
    print(f"Training executing on hardware engine target: {config.DEVICE}")

    #  Fetch data channels

    train_loader, val_loader, test_loader =  get_data_loaders(seed)

    #  Instantiate network block architecture

    model = IrisPyTorchFFN(
        input_dim= config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM ,
        output_dim=config.OUTPUT_DIM, 
        dropout_rate=config.DROPOUT_RATE
        ).to(config.DEVICE)
    
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    

    # Learning Rate Scheduler (Reduces LR by half if validation loss stalls for 5 epochs)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer= optimizer , mode='min' , factor=0.5 , patience=5 
    )

    # Early Stopping & Checkpointing Trackers
    best_val_loss =float('inf') 
    patience_counter = 0

 
    for epoch in range(config.EPOCHS):
        # -------- TRAIN --------
        model.train() 
        train_loss= 0.0

        for X_batch , y_batch in train_loader:
            X_batch, y_batch = X_batch.to(config.DEVICE) , y_batch.to(config.DEVICE)

            optimizer.zero_grad()
            prediction = model(X_batch)
            loss = criterion(prediction, y_batch)
            loss.backward() 
            optimizer.step()
           
            train_loss += loss.item() * X_batch.size(0)

        train_loss  = train_loss/len(train_loader.dataset)
        
        # -------- VALIDATION --------
        
        model.eval()
        val_loss= 0.0

        with torch.no_grad() :
            for X_val , y_val in val_loader:
                X_val, y_val = X_val.to(config.DEVICE) , y_val.to(config.DEVICE)  
                out = model(X_val)
                avg_loss = criterion(out,y_val) 

                val_loss += avg_loss.item() * X_val.size(0) 

        val_loss /= len(val_loader.dataset) 
        scheduler.step(val_loss)

        # if(epoch+1) % 10 ==0 or epoch==0 :
        #     print(f"Epoch [{epoch+1}/{config.EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")


        # -------- EARLY STOPPING --------

        if val_loss < best_val_loss :
            best_val_loss = val_loss
            patience_counter =0 #
            # Save the absolute best weights found so far (Checkpointing)
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)

        else:
            patience_counter += 1 # No improvement, increment panic counter 

        if patience_counter >= config.PATIENCE:
            print(f"Early Stopping triggered at epoch {epoch+1}! Validation loss stalled.")
            break
    print("\n--- Training Complete. Loading best checkpoint ")
    return model, test_loader
