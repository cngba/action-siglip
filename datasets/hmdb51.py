import os
import glob
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

class HMDB51VideoDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        annotation_dir: str,  # Trỏ thẳng tới thư mục testTrainMulti_7030_splits
        processor,
        split: int = 1,
        mode: str = 'train',
        num_frames: int = 8,
        prompt_template: str = "A video of a person performing {}",
        allowed_classes: list = None
    ):
        self.prompt_template = prompt_template
        self.base_dir = base_dir
        self.annotation_dir = annotation_dir
        self.processor = processor
        self.mode = mode if mode in ['train', 'val', 'test'] else 'train'
        self.num_frames = num_frames
        self.split = split
        self.allowed_classes = allowed_classes
        
        # Đọc 51 class từ file
        self.classes = self._get_classes()
        
        # BỘ LỌC ZSL
        self.unique_labels = sorted(self.classes)
        if self.allowed_classes is not None and len(self.allowed_classes) > 0:
            self.unique_labels = sorted([c for c in self.unique_labels if c in self.allowed_classes])
            
        # Load Video
        self.video_list = self._load_split()
        
        # Pipeline ảnh
        image_processor = getattr(processor, "image_processor", processor)
        mean = getattr(image_processor, "image_mean", [0.5, 0.5, 0.5])
        std = getattr(image_processor, "image_std", [0.5, 0.5, 0.5])

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

    def _get_classes(self):
        split_files = glob.glob(os.path.join(self.annotation_dir, f"*_test_split{self.split}.txt"))
        classes = sorted(list(set([os.path.basename(f).replace(f"_test_split{self.split}.txt", "") for f in split_files])))
        return classes

    def _load_split(self):
        video_list = []
        # HMDB51: 1 là train, 2 là test
        target_tag = "2" if self.mode == "test" else "1"
        
        for cls_idx, cls_name in enumerate(self.classes):
            # LỌC ZSL
            if self.allowed_classes and cls_name not in self.allowed_classes:
                continue
                
            split_file = os.path.join(self.annotation_dir, f"{cls_name}_test_split{self.split}.txt")
            if not os.path.exists(split_file):
                continue
                
            with open(split_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        vid_name, tag = parts
                        if tag == target_tag:
                            video_list.append((os.path.join(cls_name, vid_name), cls_idx))
                            
        # Chia 80/20 cho train/val
        if self.mode in ['train', 'val']:
            rng = random.Random(42)
            rng.shuffle(video_list)
            split_idx = int(len(video_list) * 0.8)
            
            if self.mode == 'train':
                video_list = video_list[:split_idx]
            elif self.mode == 'val':
                video_list = video_list[split_idx:]

        logger.info(f"Loaded {len(video_list)} videos for HMDB51 mode '{self.mode}' split {self.split}")
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
            idx = random.randint(start, max(start, end - 1)) if self.mode == 'train' else start + (end - start) // 2
            indices.append(idx)
        return np.array(indices)

    def __getitem__(self, idx):
        vid_path, label_id = self.video_list[idx]
        label_str = self.classes[label_id]
        
        # Xử lý đường dẫn
        video_path = os.path.join(self.base_dir, vid_path)
        
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            frame_indices = self._get_frame_indices(len(vr))
            frames = vr.get_batch(frame_indices) 
            frames_tensor = frames.permute(0, 3, 1, 2) 
            
            pixel_values = self.train_transforms(frames_tensor) if self.mode == 'train' else self.val_transforms(frames_tensor)
        except Exception as e:
            logger.warning(f"Error reading {video_path}: {e}")
            pixel_values = torch.zeros((self.num_frames, 3, 224, 224), dtype=torch.float32)

        text_prompt = self.prompt_template.format(label_str)
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        text_inputs = tokenizer(text_prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=64)
        
        item = {
            "pixel_values": pixel_values,
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "label_id": torch.tensor(label_id, dtype=torch.long)
        }
        if "attention_mask" in text_inputs:
            item["attention_mask"] = text_inputs["attention_mask"].squeeze(0)
            
        return item