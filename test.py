import os
import sys
import argparse
import logging
import torch
from torch.utils.data import DataLoader
import decord
from tqdm import tqdm

# Prevent diagnostic network calls over HTTPS connection layers
# os.environ["HF_HUB_OFFLINE"] = "1"

from transformers import AutoProcessor
from model import Siglip2LoRATemporalBridge
from datasets import UCF101VideoDataset

decord.bridge.set_bridge('torch')


def setup_logging(output_dir):
    """Sets up simultaneous console and file logging while suppressing external verbose alerts."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "evaluation_log.txt")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Suppress verbose core framework internal tracking logs
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    return log_file


def main(args):
    log_file_path = setup_logging(args.output_dir)
    logging.info(f"Logging initialized. Evaluation output path: {log_file_path}")

    kaggle_root = "C:\\Users\\CONG\\.cache\\kagglehub\\datasets\\matthewjansen\\ucf101-action-recognition\\versions\\4"
    base_dir = kaggle_root
    annotation_dir = kaggle_root

    if not (os.path.exists(base_dir) and os.path.exists(annotation_dir)):
        err_msg = f"Dataset folder could not be found at: {base_dir}"
        logging.error(err_msg)
        raise FileNotFoundError(err_msg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Evaluation target device context: {device}")
    
    model_name = "google/siglip2-base-patch16-224"
    processor = AutoProcessor.from_pretrained(model_name)
    
    logging.info("Constructing validation split data loader configurations...")
    val_dataset = UCF101VideoDataset(
        base_dir=base_dir, 
        annotation_dir=annotation_dir, 
        processor=processor, 
        num_frames=args.num_frames, 
        mode='val'
    )

    logging.info(f"Validation dataset split verification complete. Size: {len(val_dataset)}")
    num_workers = 0 if os.name == 'nt' else 4

    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available()
    )
    class_list = val_dataset.unique_labels 

    # Rebuild Level 4 Architecture
    model = Siglip2LoRATemporalBridge(
        model_name=model_name, 
        class_names=class_list,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    ).to(device)        

    # Safely unpack the saved checkpoint metrics file
    if args.resume and os.path.isfile(args.resume):
        logging.info(f"Loading trained weights configuration from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info("Checkpoint state metrics parsed successfully.")
    else:
        logging.warning("No checkpoint file selected. Evaluating randomized/untrained weight baselines.")

    logging.info("Beginning batch metric inference loops...")
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
            correct_5 += (top5_predicted == labels.view(-1, 1)).sum().item()
        
            progress_bar.set_postfix({
                "Top-1": f"{100 * correct_1 / total:.2f}%",
                "Top-5": f"{100 * correct_5 / total:.2f}%"
            })
    
    if total == 0:
        logging.error("No valid evaluation records processed.")
    else:
        val_acc1 = 100 * correct_1 / total
        val_acc5 = 100 * correct_5 / total
        logging.info("\n=======================================================")
        logging.info(f"Final Level 4 LoRA + Temporal Bridge Evaluation Results:")
        logging.info(f" -> Total Action Clips  : {total}")
        logging.info(f" -> Top-1 Accuracy Score : {val_acc1:.2f}% ({correct_1}/{total})")
        logging.info(f" -> Top-5 Accuracy Score : {val_acc5:.2f}% ({correct_5}/{total})")
        logging.info("=======================================================")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description="Level 4 Test Suite")
    parser.add_argument("--resume", type=str, default="./checkpoints/best_model.pt", help="Path to your best trained level 4 checkpoint parameters")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./logs")
    
    # Must mirror the training configurations exactly for linear alignment mapping shapes
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=8.0)
    
    args = parser.parse_args()
    main(args)