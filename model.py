import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipModel, AutoProcessor

class Siglip2ZeroShotBaseline(nn.Module):
    """
    Level 1: The Zero-Shot Baseline (The "Silly" Baseline)
    Vision: Frozen SigLIP. Frames are average-pooled. No trainable layers.
    Text: Frozen text encoder with a rigid, manual prompt ("A video of {class}").
    """
    def __init__(self, model_name="google/siglip2-base-patch16-224", class_names=None):
        super().__init__()
        if class_names is None:
            raise ValueError("You must provide a list of class_names for zero-shot classification.")
        else: print("Class names received.")

        self.class_names = class_names
        
        # Load the full SigLIP model (contains both vision and text towers)
        self.model = SiglipModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # ABSOLUTELY FREEZE EVERYTHING (No trainable parameters)
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.eval()  # Set to evaluation mode
        
        # Pre-prepare the rigid text prompts
        self.prompts = [f"A video of {cls_name}" for cls_name in class_names]
        self._text_features = None  # Cached text features

    def _get_text_features(self, device):
        """Encodes the manual prompts once and caches them."""
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
            
        # Extract features
        text_outputs = self.model.get_text_features(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        
        # --- SAFE TENSOR EXTRACTION ---
        if hasattr(text_outputs, "pooler_output"):
            text_features = text_outputs.pooler_output
        elif hasattr(text_outputs, "text_embeds"):
            text_features = text_outputs.text_embeds
        else:
            text_features = text_outputs
        
        text_features = F.normalize(text_features, p=2, dim=-1)
        self._text_features = text_features
        return self._text_features

    def forward(self, pixel_values):
        B, T, C, H, W = pixel_values.shape
        device = pixel_values.device
        
        # 1. Flatten B and T for the 2D encoder
        pixel_values = pixel_values.view(B * T, C, H, W)
        
        # 2. Extract features
        vision_outputs = self.model.get_image_features(pixel_values=pixel_values)
        
        # --- SAFE TENSOR EXTRACTION ---
        if hasattr(vision_outputs, "pooler_output"):
            frame_features = vision_outputs.pooler_output
        elif hasattr(vision_outputs, "image_embeds"):
            frame_features = vision_outputs.image_embeds
        else:
            frame_features = vision_outputs

        # 3. Late Fusion: Unflatten and Mean Pool across time (T)
        frame_features = frame_features.reshape(B, T, -1)  # (B, T, D)
        video_features = frame_features.mean(dim=1)     # (B, D)
        
        # 4. L2 Normalize video features
        video_features = F.normalize(video_features, p=2, dim=-1)
        
        # 5. Fetch normalized frozen text features
        text_features = self._get_text_features(device)  # (Num_Classes, D)
        
        # 6. Compute Logits
        logit_scale = self.model.logit_scale.exp()
        logits = (video_features @ text_features.T) * logit_scale  # (B, Num_Classes)
        
        return logits