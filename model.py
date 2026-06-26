import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipModel, AutoProcessor

class Siglip2LinearProbeBaseline(nn.Module):
    """
    Level 2: Language-Guided Linear Probe (The "Sober" Baseline)
    Vision: Frozen SigLIP + Trainable Temporal Module + Trainable Linear Layer.
    Text: Frozen text encoder with a rigid, manual prompt ("A video of {class}").
    Purpose: Learns a visual projection matrix and temporal sequence dynamics that adapt 
             the video embeddings to align optimally with the frozen text encoder's space.
    """
    def __init__(self, model_name="google/siglip2-base-patch16-224", class_names=None):
        super().__init__()
        if class_names is None:
            raise ValueError("You must provide a list of class_names.")
        
        self.class_names = class_names
        
        # 1. Load full SigLIP model
        self.model = SiglipModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # 2. FREEZE the entire base model (Both Vision and Text towers)
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.eval()
        
        # 3. Extract embedding dimension safely to eliminate Pylance attribute access errors
        if hasattr(self.model.config, "text_config") and self.model.config.text_config is not None:
            if isinstance(self.model.config.text_config, dict):
                embedding_dim = self.model.config.text_config.get("hidden_size", 768)
            else:
                embedding_dim = getattr(self.model.config.text_config, "hidden_size", 768)
        else:
            embedding_dim = getattr(self.model.config, "hidden_size", 768)

        # 4. FULLY TRAINABLE TEMPORAL MODULE (Added as requested)
        # Captures frame-by-frame structural motions and temporal sequence changes over time
        self.temporal_module = nn.Sequential(
            nn.Conv1d(
                in_channels=embedding_dim, 
                out_channels=embedding_dim, 
                kernel_size=3, 
                stride=1, 
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim)
        )
        
        # 5. TRAINABLE ADAPTER: Maps fused visual representations smoothly into text target matrices
        self.video_adapter = nn.Linear(embedding_dim, embedding_dim)
        
        # Pre-prepare rigid text prompts and cache placeholder structures
        self.prompts = [f"A video of {cls_name}" for cls_name in class_names]
        self._text_features = None 

    def _get_text_features(self, device):
        """Encodes the manual prompts once and caches them securely."""
        if self._text_features is not None:
            return self._text_features
        
        inputs = self.processor.tokenizer(
            text=self.prompts, 
            padding="longest", 
            truncation=True, 
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        text_outputs = self.model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output  # (Num_Classes, D)
        
        # Normalize text features
        text_features = F.normalize(text_features, p=2, dim=-1)
        self._text_features = text_features
        return self._text_features

    def forward(self, pixel_values):
        # pixel_values shape: (B, T, C, H, W)
        B, T, C, H, W = pixel_values.shape
        device = pixel_values.device
        
        # 1. Flatten B and T to map sequential frames through the 2D vision encoder
        pixel_values = pixel_values.view(B * T, C, H, W)
        
        # 2. Extract spatial sequence maps from the vision model backbone
        vision_outputs = self.model.vision_model(pixel_values=pixel_values)
        spatial_sequence = vision_outputs.last_hidden_state  # (B*T, S, D)
        
        # 3. Rebuild sequential feature dimension alignments
        _, S, D = spatial_sequence.shape
        spatial_sequence = spatial_sequence.view(B, T, S, D)
        
        # 4. Pass spatial patches through standard pooled vision attention heads
        flattened_spatial = spatial_sequence.view(B * T, S, D)
        pooled_frames = self.model.vision_model.head(flattened_spatial)  # (B*T, D)
        pooled_frames = pooled_frames.view(B, T, D)  # (B, T, D)
        
        # 5. TEMPORAL MODULE PROCESSING (Fully Trainable):
        # Transpose to (B, D, T) to satisfy PyTorch channel configurations for 1D convolutions
        temporal_input = pooled_frames.transpose(1, 2)  # (B, D, T)
        temporal_output = self.temporal_module(temporal_input)  # (B, D, T)
        
        # Pool across the remaining temporal elements (T)
        fused_video_features = temporal_output.mean(dim=-1)  # (B, D)
        
        # 6. Pass through trainable task adapter
        video_features = self.video_adapter(fused_video_features)  # (B, D)
        
        # 7. L2 Normalize video representations
        video_features = F.normalize(video_features, p=2, dim=-1)
        
        # 8. Fetch frozen text target features
        text_features = self._get_text_features(device)  # (Num_Classes, D)
        
        # 9. Compute contrastive logits output using the SigLIP exponential scalar multiplier
        logit_scale = self.model.logit_scale.exp()
        logits = (video_features @ text_features.T) * logit_scale  # (B, Num_Classes)
        
        return logits