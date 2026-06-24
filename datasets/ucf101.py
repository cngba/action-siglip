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
    Custom Dataset for UCF101 Video Action Recognition.
    Reads videos using decord and generates SigLIP text prompts.
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
        self.split = split
        self.mode = mode
        self.num_frames = num_frames
        
        self.class_to_id = self._load_class_mapping()
        self.unique_labels = sorted(list(self.class_to_id.keys()))
        
        self.label_to_id = {label: i for i, label in enumerate(self.unique_labels)}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}
        
        self.video_list = self._load_split_list()

    def _load_class_mapping(self):
        classInd_path = os.path.join(self.annotation_dir, 'classInd.txt')
        class_to_id = {}
        with open(classInd_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    class_id = int(parts[0]) - 1
                    class_name = parts[1]
                    class_to_id[class_name] = class_id
        return class_to_id

    def _load_split_list(self):
        video_list = []
        if self.mode == 'train':
            list_file = os.path.join(self.annotation_dir, f'trainlist0{self.split}.txt')
            with open(list_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        vid_path = parts[0]
                        class_name = vid_path.split('/')[0]
                        if class_name in self.class_to_id:
                            label_id = self.label_to_id[class_name]
                            video_list.append((vid_path, label_id))
        else:
            list_file = os.path.join(self.annotation_dir, f'testlist0{self.split}.txt')
            with open(list_file, 'r') as f:
                for line in f:
                    vid_path = line.strip()
                    if vid_path:
                        class_name = vid_path.split('/')[0]
                        if class_name in self.class_to_id:
                            label_id = self.label_to_id[class_name]
                            video_list.append((vid_path, label_id))
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
        video_path = os.path.join(self.base_dir, vid_path)
        
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            frame_indices = self._get_frame_indices(total_frames)
            frames = vr.get_batch(frame_indices)
            frames_np = [frame.numpy() for frame in frames]
        except Exception as e:
            # Generate actual dummy image numpy structures to prevent decord validation prints
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