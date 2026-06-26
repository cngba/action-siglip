import os
import sys
import argparse
import logging
import torch
from torch.utils.data import DataLoader
import decord
from tqdm import tqdm

from transformers import AutoProcessor
from model import Siglip2LinearProbeBaseline
from datasets import UCF101VideoDataset

decord.bridge.set_bridge('torch')


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "evaluation_log.txt")
    
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file


def main(args):
    log_file_path = setup_logging(args.output_dir)
    logging.info(f"Logging initialized. All metrics saved to: {log_file_path}")

    kaggle_root = "C:\\Users\\CONG\\.cache\\kagglehub\\datasets\\matthewjansen\\ucf101-action-recognition\\versions\\4"
    base_dir = kaggle_root
    annotation_dir = kaggle_root

    if not (os.path.exists(base_dir) and os.path.exists(annotation_dir)):
        err_msg = f"Dataset paths could not be verified at: {base_dir}"
        logging.error(err_msg)
        raise FileNotFoundError(err_msg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using evaluation device: {device}")
    
    model_name = "google/siglip2-base-patch16-224"
    processor = AutoProcessor.from_pretrained(model_name)
    
    logging.info("Setting up Evaluation Dataset and DataLoader...")
    val_dataset = UCF101VideoDataset(base_dir=base_dir, annotation_dir=annotation_dir, processor=processor, num_frames=args.num_frames, mode='val')

    logging.info(f"Validation Dataset loaded. Size: {len(val_dataset)}")
    num_workers = 0 if os.name == 'nt' else 4

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    class_list = val_dataset.unique_labels 

    model = Siglip2LinearProbeBaseline(model_name=model_name, class_names=class_list).to(device)        

    if args.resume and os.path.isfile(args.resume):
        logging.info(f"Loading trained weights from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info("Weights successfully loaded.")
    else:
        logging.warning("No valid checkpoint specified. Running raw random weights baseline!")

    logging.info("Starting Evaluation Loop...")
    model.eval()
    correct_1 = 0
    correct_5 = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Evaluation Pipeline", file=sys.stdout)
        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label_id"].to(device)
        
            logits = model(pixel_values)
            total += labels.size(0)
            
            # --- Top-1 Acc ---
            _, predicted = torch.max(logits, 1)
            correct_1 += (predicted == labels).sum().item()
            
            # --- Top-5 Acc ---
            _, top5_predicted = torch.topk(logits, k=5, dim=1)
            # Reshape label vector to match dimensions for alignment comparisons
            correct_5 += (top5_predicted == labels.view(-1, 1)).sum().item()
        
            progress_bar.set_postfix({
                "Top-1": f"{100 * correct_1 / total:.2f}%",
                "Top-5": f"{100 * correct_5 / total:.2f}%"
            })
    
    if total == 0:
        logging.error("No evaluation samples were processed.")
    else:
        val_acc1 = 100 * correct_1 / total
        val_acc5 = 100 * correct_5 / total
        logging.info("\n=======================================================")
        logging.info(f"Final Level 2 Evaluation Summary:")
        logging.info(f" -> Total Evaluated Clips: {total}")
        logging.info(f" -> Top-1 Accuracy       : {val_acc1:.2f}% ({correct_1}/{total})")
        logging.info(f" -> Top-5 Accuracy       : {val_acc5:.2f}% ({correct_5}/{total})")
        logging.info("=======================================================")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description="Level 2 Evaluation")
    parser.add_argument("--resume", type=str, default="./checkpoints/best_model.pt")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./logs")
    
    args = parser.parse_args()
    main(args)