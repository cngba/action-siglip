# /root/action-siglip/datasets/ucf101.py
import os
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
    Custom Dataset for UCF101 Video Action Recognition adapted for official TXT splits.
    """
    def __init__(
        self,
        base_dir: str,
        annotation_dir: str,
        processor,
        split: int = 1,  
        mode: str = 'train',
        num_frames: int = 8,
    ):
        self.base_dir = base_dir
        self.annotation_dir = annotation_dir
        self.processor = processor
        self.mode = mode if mode in ['train', 'val', 'test'] else 'train'
        self.num_frames = num_frames
        self.split = split
        
        # Build 0-indexed class mappings from classInd.txt
        self.label_to_id, self.id_to_label = self._build_class_mappings()
        self.unique_labels = sorted(list(self.label_to_id.keys()))
        
        # Load standard UCF101 splits
        self.video_list = self._load_split()

    def _build_class_mappings(self):
        """Scans classInd.txt to build mapping of class strings to 0-indexed IDs."""
        class_ind_path = os.path.join(self.annotation_dir, 'classInd.txt')
        label_to_id = {}
        id_to_label = {}
        
        if not os.path.exists(class_ind_path):
            raise FileNotFoundError(f"Missing class definitions at: {class_ind_path}")

        with open(class_ind_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    # UCF101 standard is 1-indexed. PyTorch requires 0-indexed.
                    idx = int(parts[0]) - 1
                    label = parts[1]
                    label_to_id[label] = idx
                    id_to_label[idx] = label
                    
        return label_to_id, id_to_label

    def _load_split(self):
        """Loads trainlist or testlist txt files based on mode."""
        video_list = []
        prefix = 'train' if self.mode == 'train' else 'test'
        list_file = os.path.join(self.annotation_dir, f'{prefix}list0{self.split}.txt')
        
        if not os.path.exists(list_file):
            raise FileNotFoundError(f"Missing split file at: {list_file}")

        with open(list_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Split handles train lists (which have spaces and IDs) and test lists (which don't)
                parts = line.split()
                vid_path = parts[0]
                
                # Extract the class name from the folder path (e.g., ApplyEyeMakeup/...)
                class_name = vid_path.split('/')[0]
                
                if class_name in self.label_to_id:
                    label_id = self.label_to_id[class_name]
                    video_list.append((vid_path, label_id))

        logger.info(f"Loaded {len(video_list)} videos for mode '{self.mode}' from split {self.split}.")
        return video_list
    
    def __len__(self):
        return len(self.video_list)

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
        
        clean_vid_path = vid_path.lstrip('/').lstrip('\\')
        clean_vid_path = clean_vid_path.replace('/', os.sep).replace('\\', os.sep)
        video_path = os.path.join(self.base_dir, clean_vid_path)
        
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            frame_indices = self._get_frame_indices(total_frames)
            frames = vr.get_batch(frame_indices)
            frames_np = [frame.numpy() for frame in frames]
        except Exception as e:
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