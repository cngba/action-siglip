import os
import csv
import torch
import random
import numpy as np
import logging
from torch.utils.data import Dataset
from decord import VideoReader, cpu
import decord

decord.bridge.set_bridge('torch')
logger = logging.getLogger(__name__)

class UCF101VideoDataset(Dataset):
    """
    Custom Dataset for UCF101 Video Action Recognition adapted for Kaggle CSV format.
    Reads videos using decord and generates SigLIP text prompts.
    """
    def __init__(
        self,
        base_dir: str,
        annotation_dir: str,
        processor,
        split: int = 1,  # Note: split is largely ignored here as Kaggle provides a fixed train/val/test split
        mode: str = 'train',
        num_frames: int = 8,
    ):
        self.base_dir = base_dir
        self.annotation_dir = annotation_dir
        self.processor = processor
        self.mode = mode if mode in ['train', 'val', 'test'] else 'train'
        self.num_frames = num_frames
        
        # Determine the correct CSV file based on the mode
        self.csv_path = os.path.join(self.annotation_dir, f'{self.mode}.csv')
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Expected CSV file not found at: {self.csv_path}")

        # Build class mappings dynamically from the CSV files
        self.label_to_id, self.id_to_label = self._build_class_mappings()
        self.unique_labels = sorted(list(self.label_to_id.keys()))
        
        # Load the list of videos for the current split/mode
        self.video_list = self._load_split_list()

    def _build_class_mappings(self):
        """Scans the CSV files to build a deterministic mapping of class strings to IDs."""
        all_classes = set()
        # Scan train, val, and test CSVs if they exist to ensure consistent class numbering
        for mode in ['train', 'val', 'test']:
            csv_p = os.path.join(self.annotation_dir, f'{mode}.csv')
            if os.path.exists(csv_p):
                with open(csv_p, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Handles common headers: 'label', 'class', or 'classname'
                        label = row.get('label') or row.get('class') or row.get('classname')
                        if label:
                            all_classes.add(label.strip())
                            
        sorted_classes = sorted(list(all_classes))
        label_to_id = {label: i for i, label in enumerate(sorted_classes)}
        id_to_label = {i: label for label, i in label_to_id.items()}
        
        if not label_to_id:
            logger.warning("No classes discovered in the CSV files. Checking folder names instead.")
        
        return label_to_id, id_to_label

    def _load_split_list(self):
        """Loads video paths and maps their labels from the target CSV file."""
        video_list = []
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Added row.get('clip_path') to perfectly match your Kaggle CSV headers
                vid_path = (row.get('clip_path') or 
                            row.get('path') or 
                            row.get('filename') or 
                            row.get('video_path'))
                
                label_str = row.get('label') or row.get('class') or row.get('classname')
                
                if vid_path and label_str:
                    vid_path = vid_path.strip()
                    label_str = label_str.strip()
                    
                    if label_str in self.label_to_id:
                        label_id = self.label_to_id[label_str]
                        video_list.append((vid_path, label_id))
                        
        logger.info(f"Successfully loaded {len(video_list)} videos for mode: {self.mode}")
        return video_list
    
    def __len__(self):
        return len(self.video_list)

    def _get_num_classes(self):
        return len(self.unique_labels)

    def _get_frame_indices(self, total_frames: int):
        if total_frames <= self.num_frames:
            return np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        seg_size = total_frames / self.num_frames
        indices = []
        for i in range(self.num_frames):
            start = int(i * seg_size)
            end = int((i + 1) * seg_size)
            if self.mode == 'train':
                idx = random.randint(start, max(start, end - 1))
            else:
                idx = start + (end - start) // 2
            indices.append(idx)
        return np.array(indices)

    def __getitem__(self, idx):
        vid_path, label_id = self.video_list[idx]
        label_str = self.id_to_label[label_id]
        
        # 1. Clean the path strings and strip any leading slashes/backslashes
        clean_vid_path = vid_path.lstrip('/').lstrip('\\')
        
        # 2. Convert slashes to match Windows/Linux OS and combine with base_dir
        clean_vid_path = clean_vid_path.replace('/', os.sep).replace('\\', os.sep)
        video_path = os.path.join(self.base_dir, clean_vid_path)
        
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            frame_indices = self._get_frame_indices(total_frames)
            frames = vr.get_batch(frame_indices)
            frames_np = [frame.numpy() for frame in frames]
        except Exception as e:
            # Generate dummy frame structure if video loading fails
            frames_np = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.num_frames)]

        text_prompt = f"A video of a person performing {label_str}"
        
        inputs = self.processor(
            images=frames_np, 
            text=text_prompt, 
            return_tensors="pt", 
            padding="max_length",
            truncation=True,
            max_length=64
        )
        
        pixel_values = inputs["pixel_values"]
        if pixel_values.dim() == 5 and pixel_values.shape[0] == 1:
            pixel_values = pixel_values.squeeze(0)
            
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0) if "attention_mask" in inputs else None

        item = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "label_id": torch.tensor(label_id, dtype=torch.long)
        }
        if attention_mask is not None:
            item["attention_mask"] = attention_mask
            
        return item