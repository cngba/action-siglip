import os
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob

try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    import cv2
    DECORD_AVAILABLE = False
    print("Warning: decord is not available. Falling back to cv2. For better performance, install decord (pip install decord).")

class UCF11VideoDataset(Dataset):
    """
    Dataset loader cho UCF11/UCF50.
    Đọc video, trích xuất num_frames khung hình, và tiền xử lý với SigLIP Processor.
    """
    def __init__(self, data_dir, processor, num_frames=8, train=True):
        self.processor = processor
        self.num_frames = num_frames
        
        if not data_dir or not os.path.exists(data_dir):
            raise ValueError(f"Đường dẫn dataset không hợp lệ: {data_dir}")
            
        # Đọc tất cả các file .avi hoặc .mpg trong các thư mục con
        self.video_paths = glob(os.path.join(data_dir, "*", "*.avi")) + glob(os.path.join(data_dir, "*", "*.mpg"))
        if len(self.video_paths) == 0:
            # Thử thư mục cấp sâu hơn nếu giải nén có cấu trúc khác
            self.video_paths = glob(os.path.join(data_dir, "*", "*", "*.avi")) + glob(os.path.join(data_dir, "*", "*", "*.mpg"))
            
        if len(self.video_paths) == 0:
            raise ValueError(f"Không tìm thấy video nào ở {data_dir}. Hãy kiểm tra lại thư mục.")
            
        # Lấy tên các lớp từ tên thư mục chứa video
        self.classes = sorted(list(set([os.path.basename(os.path.dirname(os.path.dirname(p))) if os.path.basename(os.path.dirname(p)).startswith('v_') else os.path.basename(os.path.dirname(p)) for p in self.video_paths])))
        if len(self.classes) == 0 or len(self.classes) > 100:
             self.classes = sorted(list(set([os.path.basename(os.path.dirname(os.path.dirname(p))) for p in self.video_paths])))

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Phân chia Train/Test theo Group (Tránh Data Leakage)
        from collections import defaultdict
        group_to_paths = defaultdict(list)
        for p in self.video_paths:
            group_name = os.path.dirname(p)
            group_to_paths[group_name].append(p)
            
        groups = sorted(list(group_to_paths.keys()))
        np.random.seed(42)
        np.random.shuffle(groups)
        
        split_idx = int(len(groups) * 0.8)
        if train:
            selected_groups = groups[:split_idx]
        else:
            selected_groups = groups[split_idx:]
            
        self.video_paths = []
        for g in selected_groups:
            self.video_paths.extend(group_to_paths[g])
            
        print(f"Khởi tạo dataset: {len(self.video_paths)} videos (Train={train}). Số nhóm: {len(selected_groups)}/{len(groups)}. Số lớp: {len(self.classes)}")

    def _extract_frames_decord(self, path):
        vr = VideoReader(path, ctx=cpu(0))
        total_frames = len(vr)
        # Trích xuất ngẫu nhiên hoặc cách đều
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        return frames # (T, H, W, C) numpy array

    def _extract_frames_cv2(self, path):
        import cv2
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 100 # Fallback safety
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        current_frame = 0
        idx_pointer = 0
        
        while cap.isOpened() and idx_pointer < len(indices):
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame == indices[idx_pointer]:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                idx_pointer += 1
            current_frame += 1
        cap.release()
        
        # Padding nếu video quá ngắn hoặc lỗi đọc
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if len(frames) > 0 else np.zeros((224, 224, 3), dtype=np.uint8))
            
        return np.array(frames)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        
        # Xác định nhãn từ đường dẫn
        parts = path.split(os.sep)
        label_str = next((p for p in parts if p in self.classes), parts[-3])
        label = self.class_to_idx[label_str]
        
        try:
            if DECORD_AVAILABLE:
                try:
                    frames = self._extract_frames_decord(path)
                except Exception as decord_e:
                    # Fallback to cv2 if decord encounters broken metadata
                    frames = self._extract_frames_cv2(path)
            else:
                frames = self._extract_frames_cv2(path)
                
            # Đảm bảo frames là list của PIL Image hoặc list numpy array 3D
            frames_list = [frame for frame in frames]
            
            # Tiền xử lý với SigLIP Processor
            inputs = self.processor(images=frames_list, return_tensors="pt")
            pixel_values = inputs["pixel_values"] # (T, C, H, W)
            
            return pixel_values, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            # Xử lý video bị lỗi (corrupted)
            print(f"Lỗi khi đọc video {path}: {e}. Trả về một frame padding ngẫu nhiên.")
            dummy_pixel_values = torch.zeros((self.num_frames, 3, 224, 224))
            return dummy_pixel_values, torch.tensor(label, dtype=torch.long)
