# action-siglip/datasets/ucf101.py
import os
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


class UCF101VideoDataset(Dataset):
    """
    Optimized Dataset for UCF101 using pure PyTorch Tensor transformations.
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
        
        # Step 1: Build the class-name to label-id mapping used by the dataset.
        self.label_to_id, self.id_to_label = self._build_class_mappings()
        self.unique_labels = sorted(list(self.label_to_id.keys()))

        # Step 2: Load all video paths for the requested split and mode.
        self.video_list = self._load_split()
        
        # Step 3: Read normalization statistics from the processor so frames are standardized consistently.
        image_processor = getattr(processor, "image_processor", processor)
        mean = getattr(image_processor, "image_mean", [0.5, 0.5, 0.5])
        std = getattr(image_processor, "image_std", [0.5, 0.5, 0.5])

        # Step 4: Create separate augmentation pipelines for training and validation/inference.
        self.train_transforms = v2.Compose([
            v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True), # Scales uint8 [0, 255] to float32 [0.0, 1.0]
            v2.Normalize(mean=mean, std=std),
        ])
        
        self.val_transforms = v2.Compose([
            v2.Resize(256, antialias=True),
            v2.CenterCrop((224, 224)),
            v2.ToDtype(torch.float32, scale=True), # Scales uint8 [0, 255] to float32 [0.0, 1.0]
            v2.Normalize(mean=mean, std=std),
        ])

    def _build_class_mappings(self):
        # Read the UCF101 class index file so each action category can be mapped to an integer label.
        class_ind_path = os.path.join(self.annotation_dir, 'classInd.txt')
        label_to_id, id_to_label = {}, {}
        
        if not os.path.exists(class_ind_path):
            raise FileNotFoundError(f"Missing class definitions at: {class_ind_path}")

        with open(class_ind_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    idx = int(parts[0]) - 1
                    label = parts[1]
                    label_to_id[label] = idx
                    id_to_label[idx] = label
                    
        return label_to_id, id_to_label

    def _load_split(self):
        # Collect the video paths and labels listed in the chosen annotation split file.
        video_list = []
        prefix = 'train' if self.mode == 'train' else 'test'
        list_file = os.path.join(self.annotation_dir, f'{prefix}list0{self.split}.txt')
        
        if not os.path.exists(list_file):
            raise FileNotFoundError(f"Missing split file at: {list_file}")

        with open(list_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Each line contains a video path followed by its class name.
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                vid_path = parts[0]
                class_name = vid_path.split('/')[0]
                
                if class_name in self.label_to_id:
                    label_id = self.label_to_id[class_name]
                    video_list.append((vid_path, label_id))

        logger.info(f"Loaded {len(video_list)} videos for mode '{self.mode}' from split {self.split}.")
        return video_list
    
    def __len__(self):
        return len(self.video_list)

    def _get_frame_indices(self, total_frames: int):
        # Choose a fixed set of frame positions so each video is represented consistently.
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
        # Step 1: Load the metadata for the requested video sample.
        vid_path, label_id = self.video_list[idx]
        label_str = self.id_to_label[label_id]
        
        # Step 2: Normalize the path so it works on any operating system.
        clean_vid_path = vid_path.lstrip('/').lstrip('\\')
        clean_vid_path = clean_vid_path.replace('/', os.sep).replace('\\', os.sep)
        video_path = os.path.join(self.base_dir, clean_vid_path)
        
        try:
            # Step 3: Open the video and sample a fixed number of frames.
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            frame_indices = self._get_frame_indices(total_frames)
            
            # Decord returns a tensor shaped (T, H, W, C); convert it to (T, C, H, W) for PyTorch.
            frames = vr.get_batch(frame_indices) 
            frames_tensor = frames.permute(0, 3, 1, 2) 
            
            # Step 4: Apply the selected augmentation and normalization pipeline.
            if self.mode == 'train':
                pixel_values = self.train_transforms(frames_tensor) # (T, C, 224, 224)
            else:
                pixel_values = self.val_transforms(frames_tensor)   # (T, C, 224, 224)
                
        except Exception as e:
            # Step 5: If the video cannot be read, fall back to a zero tensor with the expected shape.
            logger.warning(f"Error reading video {video_path}: {e}")
            pixel_values = torch.zeros((self.num_frames, 3, 224, 224), dtype=torch.float32)

        # Step 6: Build a simple text prompt for the action label and tokenize it.
        text_prompt = f"A video of a person performing {label_str}"
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        
        text_inputs = tokenizer(
            text_prompt, 
            return_tensors="pt", 
            padding="max_length",
            truncation=True,
            max_length=64
        )
        
        input_ids = text_inputs["input_ids"].squeeze(0)
        attention_mask = text_inputs["attention_mask"].squeeze(0) if "attention_mask" in text_inputs else None

        # Step 7: Return a dictionary in the format expected by the training pipeline.
        item = {
            "pixel_values": pixel_values, # Output is directly a (8, 3, 224, 224) float32 Tensor
            "input_ids": input_ids,
            "label_id": torch.tensor(label_id, dtype=torch.long)
        }
        if attention_mask is not None:
            item["attention_mask"] = attention_mask
            
        return item