import os
import argparse
import torch
import cv2 
import numpy as np
from torch.utils.data import DataLoader
import decord
from tqdm import tqdm

from transformers import AutoProcessor
# Level 1 uses the Zero Shot Baseline model structure
from model import Siglip2ZeroShotBaseline
from datasets import UCF101VideoDataset

decord.bridge.set_bridge('torch')


def main(args):
    # Direct path configurations pointing to your local Kaggle dataset
    kaggle_root = "C:\\Users\\CONG\\.cache\\kagglehub\\datasets\\matthewjansen\\ucf101-action-recognition\\versions\\4"
    
    # Handle both direct text-list placements and nested subfolders
    if os.path.exists(os.path.join(kaggle_root, "UCF-101")):
        base_dir = os.path.join(kaggle_root, "UCF-101")
    else:
        base_dir = kaggle_root

    annotation_dir = os.path.join(base_dir, "ucfTrainTestlist")
    if not os.path.exists(annotation_dir):
        annotation_dir = base_dir 

    # Enforce a strict path validation check right away
    if not (os.path.exists(base_dir) and os.path.exists(annotation_dir)):
        raise FileNotFoundError(
            f"Dataset paths could not be verified.\n"
            f"Looking for base directory: {base_dir}\n"
            f"Looking for annotations directory: {annotation_dir}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_name = "google/siglip2-base-patch16-224"
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_name)
    
    print("Setting up Evaluation Dataset and DataLoader...")
    val_dataset = UCF101VideoDataset(
        base_dir=base_dir, 
        annotation_dir=annotation_dir, 
        processor=processor, 
        num_frames=args.num_frames, 
        mode='val'
    )

    # Hard set to 0 on Windows ('nt') to reliably prevent multiprocessing SpawnErrors
    num_workers = 0 if os.name == 'nt' else 4
    print(f"Data pipeline configured with num_workers={num_workers}")

    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available()
    )
    print(f"Validation Dataset Size: {len(val_dataset)}")
    
    print("\nInitializing Level 1: Zero-Shot Baseline Model...")
    class_list = val_dataset.unique_labels 

    model = Siglip2ZeroShotBaseline(
        model_name=model_name, 
        class_names=class_list
    ).to(device)        

    if args.resume and os.path.isfile(args.resume):
        print(f"Loading weights from checkpoint {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

    # --- LEVEL 1: ZERO-SHOT EVALUATION ---
    print("\n--- Starting Level 1 Zero-Shot Evaluation (Silly Baseline) ---")
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Evaluation")
        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label_id"].to(device)
        
            logits = model(pixel_values)
            
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
            progress_bar.set_postfix({"Acc": f"{100 * correct / total:.2f}%"})
    
    val_acc = 100 * correct / total
    print(f"\n=======================================================")
    print(f"Final Level 1 Zero-Shot Baseline Accuracy: {val_acc:.2f}%")
    print(f"=======================================================")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description="Level 1 VideoSiglip2 Evaluation")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint if parsing pre-saved parameter layers")
    parser.add_argument("--num_frames", type=int, default=8, help="Number of frames sampled uniformly per video clip")
    parser.add_argument("--batch_size", type=int, default=4, help="Data batch size tracking constraints")
    
    args = parser.parse_args()
    main(args)