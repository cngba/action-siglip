# test.py
# Author: Cong
# Decoupled Inference Validation Script - Tracking Explicit Multi-Class Configurations
# Optimized for Level 5 Dual-Tower Co-Alignment and Hybrid Temporal Bridge Performance
# Configured for Unified Mode Matrix Parsers (ucf101.yaml)

import os
import sys
import time
import yaml
import argparse
import logging
import torch
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Prevent diagnostic network verification calls over HTTPS layers
os.environ["HF_HUB_OFFLINE"] = "1"

try:
    import wandb
except ImportError:
    wandb = None

from model import Siglip2FullLoRATemporalBridge
from datasets import UCF101VideoDataset


def setup_logging(output_dir):
    """Sets up localized file tracking logging outputs."""
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
    
    # Silence external core diagnostic network trackers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    return log_file


@torch.no_grad()
def extract_ensembled_text_features(model, class_names, processor, device):
    """
    Computes ensembled text embeddings for all 101 target action classes once.
    Averages multiple prompt contexts to improve zero-shot and PEFT generalization.
    """
    logging.info("Extracting cached ensembled text features for all target classes...")
    model.eval()
    
    # Standard academic prompt templates for action datasets
    templates = [
        "A video of a person performing {}.",
        "A video of someone {}.",
        "An action shot of a person doing {}.",
        "A crisp video recording of a human participating in {}.",
        "A dynamic video showing the action of {}."
    ]
    
    class_embeddings = []
    
    # Assuming your model has a text encoding pipeline (e.g., model.get_text_features or similar)
    # If your model architecture computes logits directly inside forward via cross-attention,
    # ensure your model code contains a method to isolate text embedding projections.
    has_text_encoder = hasattr(model, 'get_text_features') or hasattr(model, 'encode_text')

    if not has_text_encoder:
        logging.warning("Model lacks a detached text feature extraction function. Falling back to batch classification.")
        return None

    for class_name in class_names:
        # Format the label nicely (e.g., convert "PlayingBasketball" to "playing basketball")
        clean_name = ''.join([' ' + c.lower() if c.isupper() else c for c in class_name]).strip()
        prompts = [template.format(clean_name) for template in templates]
        
        # Tokenize all prompts for this class
        inputs = processor(
            text=prompts,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        ).to(device)
        
        # Extract features through the text tower
        if hasattr(model, 'get_text_features'):
            prompt_embeds = model.get_text_features(inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
        else:
            prompt_embeds = model.encode_text(inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
            
        # Normalize and average across templates
        prompt_embeds = F.normalize(prompt_embeds, p=2, dim=-1)
        ensembled_class_embed = prompt_embeds.mean(dim=0, keepdim=True)
        ensembled_class_embed = F.normalize(ensembled_class_embed, p=2, dim=-1)
        
        class_embeddings.append(ensembled_class_embed)
        
    # Stack into a final matrix of shape: [Num_Classes, Embedding_Dim]
    return torch.cat(class_embeddings, dim=0)


def validate(epoch, dataloader, model, device, cached_text_features=None):
    """Clean evaluation checking loop isolating your final target list over to wandb."""
    model.eval()
    all_labels = []
    all_preds = []
    all_logits = []
    
    start_time = time.time()
    total_videos = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validation (Epoch {epoch})", file=sys.stdout):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label_id"].to(device)
            total_videos += labels.size(0)
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                # If cached ensembled text embeddings are available, use matrix multiplication direct inference
                if cached_text_features is not None and hasattr(model, 'get_vision_features'):
                    video_features = model.get_vision_features(pixel_values)
                    video_features = F.normalize(video_features, p=2, dim=-1)
                    
                    # Compute cosine similarities scaled by the model's logit scale parameter if available
                    logit_scale = getattr(model, 'logit_scale', 100.0)
                    if isinstance(logit_scale, torch.Tensor):
                        logit_scale = logit_scale.exp()
                    logits = torch.matmul(video_features, cached_text_features.t()) * logit_scale
                else:
                    # Standard forward pass fallback using dataset text prompt extraction
                    logits = model(pixel_values)
            
            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.append(logits.cpu())
            
    end_time = time.time()
    inference_duration = end_time - start_time
    
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # 1. Top-1 Target Calculation
    top1_acc = accuracy_score(all_labels, all_preds) * 100
    
    # 2. Top-5 Target Calculation
    num_classes = all_logits.shape[1]
    k = min(5, num_classes)
    _, topk_indices = torch.topk(all_logits, k, dim=1)
    topk_correct = topk_indices.eq(torch.tensor(all_labels).view(-1, 1)).sum().item()
    top5_acc = (topk_correct / total_videos) * 100

    # 3. Macro Target Precision, Recall, and F1 calculations
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    precision_pct = precision * 100
    recall_pct = recall * 100
    f1_pct = f1 * 100

    # Console prints and logging tracking elements configurations
    report_header = "\n" + "="*50 + f"\n            DETAILED PERFORMANCE REPORT (ENSEMBLED PROMPTS)           \n" + "="*50
    print(report_header)
    print(f"Top-1 Accuracy:  {top1_acc:.2f}%")
    print(f"Top-5 Accuracy:  {top5_acc:.2f}%")
    print(f"Macro F1-Score:  {f1_pct:.2f}%")
    print(f"Macro Precision: {precision_pct:.2f}%")
    print(f"Macro Recall:    {recall_pct:.2f}%")
    print("-"*50)
    print(f"Inference Time:  {inference_duration:.2f} seconds total")
    print(f"Throughput:      {(total_videos / max(0.001, inference_duration)):.2f} clips/sec")
    print("="*50 + "\n")

    if wandb and wandb.run:
        try:
            wandb.log({
                "epoch": epoch,
                "val_top1": top1_acc,
                "val_top5": top5_acc,
                "val_f1": f1_pct,
                "val_precision": precision_pct,
                "val_recall": recall_pct
            })
        except Exception as e:
            print(f"[Wandb Telemetry Warning] Connection stream skipped: {e}")

    return {
        "top1": top1_acc,
        "top5": top5_acc,
        "f1": f1_pct
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Evaluation Target Script")
    parser.add_argument('--config', '-cfg', default='configs/ucf101.yaml', help='Path to the unified YAML file')
    parser.add_argument('--mode', '-m', default='train_full_peft', help='Target mode matrix selection from YAML')
    parser.add_argument("--weights", type=str, default="", help="Optional path to best_model.pt weights")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Missing configuration file at: {args.config}")
        
    with open(args.config, 'r') as f:
        raw_yaml = yaml.safe_load(f)

    if "modes" not in raw_yaml or args.mode not in raw_yaml["modes"]:
        raise KeyError(f"Selected target mode configuration option '{args.mode}' not discovered in the YAML map matrix.")

    config = {
        "model_name": raw_yaml.get("model_name", "google/siglip2-base-patch16-224"),
        "seed": raw_yaml.get("seed", 1024),
        "base_dir": raw_yaml["data"]["base_dir"],
        "annotation_dir": raw_yaml["data"]["annotation_dir"],
        "split": raw_yaml["data"]["split"],
        "num_frames": raw_yaml["data"]["num_segments"],
        "num_workers": raw_yaml["data"]["workers"]
    }
    
    mode_specific_config = raw_yaml["modes"][args.mode]
    config.update(mode_specific_config)
    
    setup_logging(os.path.dirname(config["log_file"]))
    logging.info(f"Loaded master configuration profile for evaluation mode: {args.mode}")

    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Targeting active processing evaluation hardware context: {device}")

    processor = AutoProcessor.from_pretrained(config["model_name"])
    
    logging.info("Constructing video datasets validation loader configurations...")
    val_dataset = UCF101VideoDataset(
        base_dir=config["base_dir"], 
        annotation_dir=config["annotation_dir"], 
        processor=processor, 
        split=config["split"],
        num_frames=config["num_frames"], 
        mode='val'
    )
    
    g = torch.Generator()
    g.manual_seed(config["seed"])

    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    num_workers = min(config["num_workers"], os.cpu_count()) if os.name != 'nt' else 0
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        generator=g
    )

    class_list = val_dataset.unique_labels
    
    model = Siglip2FullLoRATemporalBridge(
        model_name=config["model_name"], 
        class_names=class_list,
        lora_r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        use_vision_lora=config["vision_lora"],
        use_text_lora=config["text_lora"],
        use_temporal_adapter=config["temporal_module"]
    ).to(device)

    current_epoch = 0

    if args.mode == "zero_shot":
        logging.info("Executing pure Zero-Shot baseline performance evaluation over vanilla pre-trained model layers.")
    else:
        checkpoint_target = args.weights if args.weights else os.path.join(config["checkpoint_dir"], "best_model.pt")
        if checkpoint_target and os.path.isfile(checkpoint_target):
            logging.info(f"Loading fine-tuned checkpoint weights directly from: {checkpoint_target}")
            checkpoint = torch.load(checkpoint_target, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            current_epoch = checkpoint.get('epoch', 0)
        else:
            logging.warning(f"No checkpoint matched at destination: '{checkpoint_target}'. "
                            f"Evaluating vanilla pre-trained weights for fine-tuning mode structure instead.")

    # Run cache pre-extraction before evaluating the data loader stream
    cached_text_features = extract_ensembled_text_features(model, class_list, processor, device)

    if wandb:
        project_name = "action-siglip2-zeroshot-baseline" if args.mode == "zero_shot" else "action-siglip2-peft-eval"
        wandb.init(
            project=project_name,
            name=f"eval_{args.mode}_{config['num_frames']}f",
            config=config
        )

    validate(current_epoch, val_loader, model, device, cached_text_features=cached_text_features)
    
    if wandb:
        wandb.finish()