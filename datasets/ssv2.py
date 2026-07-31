import os
import json
import torch
import random
import numpy as np
import logging
from torch.utils.data import Dataset
from decord import VideoReader, cpu
import decord
from torchvision.transforms import v2

decord.bridge.set_bridge('torch')
logger = logging.getLogger(__name__)


class SSv2VideoDataset(Dataset):
    """
    Optimized Dataset for Something-Something v2 (SSv2) using PyTorch Tensor transformations.
    """
    def __init__(
        self,
        base_dir: str,
        annotation_dir: str,
        processor,
        mode: str = 'train',
        num_frames: int = 8,
        prompt_template: str = "A video of a person {}", 
        allowed_classes=None
    ):
        self.prompt_template = prompt_template
        self.allowed_classes = allowed_classes
        self.base_dir = base_dir
        self.annotation_dir = annotation_dir
        self.processor = processor
        self.mode = mode if mode in ['train', 'val', 'test'] else 'train'
        self.num_frames = num_frames
        
        # Step 1: Build the class-name to label-id mapping from labels.json
        self.label_to_id, self.id_to_label = self._build_class_mappings()
        self.unique_labels = sorted(list(self.label_to_id.keys()))
        
        if self.allowed_classes is not None and len(self.allowed_classes) > 0:
            self.unique_labels = sorted([c for c in self.unique_labels if c in self.allowed_classes])

        # Step 2: Load video metadata from the appropriate JSON file
        self.video_list = self._load_split()
        
        # Step 3: Normalization statistics from processor
        image_processor = getattr(processor, "image_processor", processor)
        mean = getattr(image_processor, "image_mean", [0.5, 0.5, 0.5])
        std = getattr(image_processor, "image_std", [0.5, 0.5, 0.5])

        # Step 4: Transformations
        self.train_transforms = v2.Compose([
            v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ])
        
        self.val_transforms = v2.Compose([
            v2.Resize(256, antialias=True),
            v2.CenterCrop((224, 224)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ])

    def _build_class_mappings(self):
        """Reads labels.json to create mappings."""
        labels_path = os.path.join(self.annotation_dir, 'labels.json')
        
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Missing labels definition at: {labels_path}")

        with open(labels_path, 'r', encoding='utf-8') as f:
            label_to_id = json.load(f)

        # Convert string values in JSON to integer IDs if needed
        label_to_id = {k: int(v) for k, v in label_to_id.items()}
        id_to_label = {v: k for k, v in label_to_id.items()}
        
        return label_to_id, id_to_label

    def _load_split(self):
        """Loads train.json, validation.json, or test.json."""
        file_map = {
            'train': 'train.json',
            'val': 'validation.json',
            'test': 'test.json'
        }
        
        json_file = os.path.join(self.annotation_dir, file_map[self.mode])
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Missing split file at: {json_file}")

        video_list = []
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for entry in data:
            video_id = entry['id']
            
            if self.mode == 'test':
                # Test set doesn't contain labels
                video_list.append((video_id, None, None))
            else:
                # SSv2 templates contain placeholders like "[something]"
                template_name = entry['template'].replace('[', '').replace(']', '')
                
                if template_name in self.label_to_id:
                    if self.allowed_classes is not None and len(self.allowed_classes) > 0:
                        if template_name not in self.allowed_classes:
                            continue

                    label_id = self.label_to_id[template_name]
                    video_list.append((video_id, label_id, template_name))

        logger.info(f"Loaded {len(video_list)} SSv2 videos for mode '{self.mode}'.")
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
        video_id, label_id, label_str = self.video_list[idx]
        
        # Build path pointing to .webm files (e.g. 196166.webm)
        video_path = os.path.join(self.base_dir, f"{video_id}.webm")
        
        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            total_frames = len(vr)
            frame_indices = self._get_frame_indices(total_frames)
            
            frames = vr.get_batch(frame_indices) 
            frames_tensor = frames.permute(0, 3, 1, 2) 
            
            if self.mode == 'train':
                pixel_values = self.train_transforms(frames_tensor)
            else:
                pixel_values = self.val_transforms(frames_tensor)
                
        except Exception as e:
            logger.warning(f"Error reading video {video_path}: {e}")
            pixel_values = torch.zeros((self.num_frames, 3, 224, 224), dtype=torch.float32)

        # Build clean string prompt replacing placeholders for the tokenizer
        clean_text = (label_str or "").replace("something", "an object")
        text_prompt = self.prompt_template.format(clean_text)
        
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        text_inputs = tokenizer(
            text_prompt, 
            return_tensors="pt", 
            padding="max_length",
            truncation=True,
            max_length=64
        )
        
        input_ids = text_inputs["input_ids"].squeeze(0)
        attention_mask = text_inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(0)

        item = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "video_id": video_id
        }

        if label_id is not None:
            item["label_id"] = torch.tensor(label_id, dtype=torch.long)
            item["label_name"] = label_str

        if attention_mask is not None:
            item["attention_mask"] = attention_mask
            
        return item