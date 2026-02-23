import os
import torch 
from utils.metrics import dice_score

def train_epochs(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_dice = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_score(logits, masks)
        
    return total_loss / len(loader), total_dice / len(loader)

def evaluate_epochs(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            total_loss += loss.item()
            total_dice += dice_score(logits, masks)
            
    return total_loss / len(loader), total_dice / len(loader)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, save_dir, epochs):
    os.makedirs(save_dir, exist_ok=True)
    best_dice = 0.0

    for epoch in range(epochs):
        train_loss, train_dice = train_epochs(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = evaluate_epochs(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Train Dice: {train_dice:.4f} - Val Loss: {val_loss:.4f} - Val Dice: {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print("Model saved!")
        
    return model, best_dice