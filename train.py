import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import decord
from tqdm import tqdm

from transformers import AutoProcessor
from model import Siglip2LinearProbeBaseline
from datasets import UCF101VideoDataset

decord.bridge.set_bridge('torch')


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    model.model.eval() # Keep backbone frozen
    
    running_loss = 0.0
    correct_1 = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["label_id"].to(device)
        
        optimizer.zero_grad()
        
        logits = model(pixel_values)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct_1 += (predicted == labels).sum().item()
        
        progress_bar.set_postfix({
            "Loss": f"{loss.item():.4f}", 
            "Top-1": f"{100 * correct_1 / total:.2f}%"
        })
        
    epoch_loss = running_loss / total
    epoch_acc1 = 100 * correct_1 / total
    return epoch_loss, epoch_acc1


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_1 = 0
    correct_5 = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label_id"].to(device)
            
            logits = model(pixel_values)
            loss = criterion(logits, labels)
            
            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            
            # Top-1 Accuracy Calculation
            _, predicted = torch.max(logits, 1)
            correct_1 += (predicted == labels).sum().item()
            
            # Top-5 Accuracy Calculation
            _, top5_predicted = torch.topk(logits, k=5, dim=1)
            correct_5 += (top5_predicted == labels.view(-1, 1)).sum().item()
            
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}", 
                "Top-1": f"{100 * correct_1 / total:.2f}%",
                "Top-5": f"{100 * correct_5 / total:.2f}%"
            })
            
    val_loss = running_loss / total
    val_acc1 = 100 * correct_1 / total
    val_acc5 = 100 * correct_5 / total
    return val_loss, val_acc1, val_acc5


def main(args):
    kaggle_root = "C:\\Users\\CONG\\.cache\\kagglehub\\datasets\\matthewjansen\\ucf101-action-recognition\\versions\\4"
    base_dir = kaggle_root
    annotation_dir = kaggle_root

    if not (os.path.exists(base_dir) and os.path.exists(annotation_dir)):
        raise FileNotFoundError(f"Dataset paths could not be verified at: {base_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_name = "google/siglip2-base-patch16-224"
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_name)
    
    print("Setting up Datasets...")
    train_dataset = UCF101VideoDataset(base_dir=base_dir, annotation_dir=annotation_dir, processor=processor, num_frames=args.num_frames, mode='train')
    val_dataset = UCF101VideoDataset(base_dir=base_dir, annotation_dir=annotation_dir, processor=processor, num_frames=args.num_frames, mode='val')

    num_workers = 0 if os.name == 'nt' else 4
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    
    class_list = train_dataset.unique_labels 
    model = Siglip2LinearProbeBaseline(model_name=model_name, class_names=class_list).to(device)        

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_acc1 = 0.0

    print("\n--- Starting Training Pipeline ---")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        train_loss, train_acc1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc1, val_acc5 = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Epoch Summary -> Train Loss: {train_loss:.4f} | Train Top-1 Acc: {train_acc1:.2f}%")
        print(f"Epoch Summary -> Val Loss: {val_loss:.4f} | Val Top-1: {val_acc1:.2f}% | Val Top-5: {val_acc5:.2f}%")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc1': val_acc1,
            'val_acc5': val_acc5,
        }
        
        torch.save(checkpoint, os.path.join(args.output_dir, "latest_checkpoint.pt"))
        if val_acc1 > best_val_acc1:
            best_val_acc1 = val_acc1
            torch.save(checkpoint, os.path.join(args.output_dir, "best_model.pt"))
            print(f"New best model saved with Top-1 accuracy: {best_val_acc1:.2f}%!")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description="Level 2 VideoSiglip2 Training")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    
    args = parser.parse_args()
    main(args)