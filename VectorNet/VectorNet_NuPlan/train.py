import os
import time
import torch
import numpy as np
import torch.optim as optim
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from utils import config
from utils.dataset import GraphDataset, GraphData
from vectornet.vectornet import VectornetGNN

def validate(model, validate_loader, device):
    model.eval()
    val_loss = 0.0
    num_samples = 0
    with torch.no_grad():
        for data in validate_loader:
            data = data.to(device)
            y = data.y.to(torch.float32).view(-1, config.OUT_CHANNELS)
            out = model(data)
            loss = F.mse_loss(out, y)
            val_loss += config.BATCH_SIZE * loss.item()
            num_samples += y.shape[0]
    model.train()
    return val_loss / num_samples

if __name__ == "__main__":
    # Set seed
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    
    # Load the entire dataset
    full_data = GraphDataset(config.TRAIN_PATH)
    
    # Split the dataset into training and validation sets
    train_indices, val_indices = train_test_split(
        np.arange(len(full_data)),
        test_size=config.VALIDATION_SPLIT,
        random_state=config.SEED
    )
    
    train_data = full_data[train_indices]
    validate_data = full_data[val_indices]
    
    # Load training data
    train_loader = DataLoader(
        train_data,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )
    
    # Load validation data
    validate_loader = DataLoader(
        validate_data,
        batch_size=config.BATCH_SIZE
    )

    # Create predictor
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = VectornetGNN(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config.DECAY_LR_EVERY, 
        gamma=config.DECAY_LR_FACTOR
    )
    
    global_step = 0
    model.train()
    
    for epoch in range(config.EPOCHS):
        print(f"Start training at epoch: {epoch}")
        
        acc_loss = 0.0
        num_samples = 0
        start_tic = time.time()
        
        for data in train_loader:
            data = data.to(device)
            y = data.y.to(torch.float32).view(-1, config.OUT_CHANNELS)
            
            optimizer.zero_grad()
            out = model(data)
            
            loss = F.mse_loss(out, y)
            loss.backward()
            
            acc_loss += config.BATCH_SIZE * loss.item()
            num_samples += y.shape[0]
            
            optimizer.step()
                        
            if (global_step + 1) % config.SHOW_EVERY == 0:
                loss_value = loss.item()
                learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
                elapsed_time = time.time() - start_tic

                # Print training info
                print(f"epoch-{epoch}, step-{global_step}: "
                      f"loss: {loss_value:.3f}, "
                      f"lr: {learning_rate:.6f}, "
                      f"time: {elapsed_time:.4f} sec")
            
            global_step += 1
            
        scheduler.step()
        
        # Print every epoch
        loss_value = acc_loss / num_samples
        learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        elapsed_time = time.time() - start_tic
        print(f"Finished epoch {epoch}: "
              f"loss: {loss_value:.3f}, "
              f"lr: {learning_rate:.6f}, "
              f"time: {elapsed_time:.4f} sec")
        
        # Validate
        val_loss = validate(model, validate_loader, device)
        print(f"Validation loss after epoch {epoch}: {val_loss:.3f}")
    
        # Save params to local
        os.makedirs(config.WEIGHT_PATH, exist_ok=True)
        model_filename = f"model_epoch_{epoch+1:03d}.pth"
        model_path = os.path.join(config.WEIGHT_PATH, model_filename)
        torch.save(model.state_dict(), model_path)