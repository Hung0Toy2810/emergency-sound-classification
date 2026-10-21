import torch
import torch.nn as nn
import torch.nn.functional as F
import os

def adapt_projection_layer(vit_model, stride=(10, 10)):
    """
    Modifies the patch embedding projection layer to accept 1-channel audio spectrograms.
    Averages the pre-trained weights across the 3 RGB channels.
    """
    old_proj = vit_model.embeddings.patch_embeddings.projection
    old_weight = old_proj.weight  # Shape: [hidden_size, 3, patch_size_h, patch_size_w]
    
    # Average the weights across the channel dimension
    new_weight = old_weight.mean(dim=1, keepdim=True)  # Shape: [hidden_size, 1, patch_size_h, patch_size_w]
    
    # Create new projection layer with 1 input channel and the custom stride
    new_proj = nn.Conv2d(
        in_channels=1,
        out_channels=old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=stride,
        padding=old_proj.padding
    )
    new_proj.weight = nn.Parameter(new_weight)
    new_proj.bias = old_proj.bias
    
    # Replace the old projection layer
    vit_model.embeddings.patch_embeddings.projection = new_proj
    
    # Bypass Hugging Face's internal channel and size checks by binding a custom forward method
    import types
    def custom_patch_embeddings_forward(self, pixel_values, interpolate_pos_encoding=False):
        return self.projection(pixel_values).flatten(2).transpose(1, 2)
        
    vit_model.embeddings.patch_embeddings.forward = types.MethodType(
        custom_patch_embeddings_forward, 
        vit_model.embeddings.patch_embeddings
    )
    
    def custom_embeddings_forward(self, pixel_values, bool_masked_pos=None, interpolate_pos_encoding=False):
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)
        
        # Add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        
        # Add positional embeddings directly without shape/size checks
        embeddings = embeddings + self.position_embeddings
        
        embeddings = self.dropout(embeddings)
        return embeddings
        
    vit_model.embeddings.forward = types.MethodType(
        custom_embeddings_forward,
        vit_model.embeddings
    )

def interpolate_pos_embeddings(vit_model, num_freq_patches=12, num_time_patches=99):
    """
    Adapts the 2D positional embeddings of a pre-trained ViT model to a new rectangular grid
    using bilinear interpolation.
    """
    old_pos = vit_model.embeddings.position_embeddings  # Shape: [1, num_old_patches + 1, hidden_size]
    cls_pos = old_pos[:, :1, :]  # CLS token positional embedding: [1, 1, hidden_size]
    patch_pos = old_pos[:, 1:, :]  # Patch positional embeddings: [1, num_old_patches, hidden_size]
    
    # Determine the old grid size (assuming square)
    H_old = W_old = int(patch_pos.shape[1] ** 0.5)
    
    # Reshape to [1, hidden_size, H_old, W_old] for 2D interpolation
    hidden_size = old_pos.shape[-1]
    patch_pos = patch_pos.transpose(1, 2).reshape(1, hidden_size, H_old, W_old)
    
    # Perform 2D bilinear interpolation to new rectangular patch grid size
    new_patch_pos = F.interpolate(
        patch_pos,
        size=(num_freq_patches, num_time_patches),
        mode='bilinear',
        align_corners=False
    )
    
    # Reshape back to 1D sequence: [1, num_freq_patches * num_time_patches, hidden_size]
    new_patch_pos = new_patch_pos.reshape(1, hidden_size, -1).transpose(1, 2)
    
    # Concatenate the CLS token positional embedding back to the start
    new_pos = torch.cat([cls_pos, new_patch_pos], dim=1)  # Shape: [1, 1 + N, hidden_size]
    
    # Update the positional embedding parameter in the model
    vit_model.embeddings.position_embeddings = nn.Parameter(new_pos)

class ViTAST(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224", num_classes=4,
                 num_freq_patches=12, num_time_patches=99):
        """
        Audio Spectrogram Transformer (AST) adapter wrapping standard Hugging Face ViT model.
        
        Args:
            model_name (str): Hugging Face pretrained ViT identifier
            num_classes (int): Number of target classes (default 4)
            num_freq_patches (int): Number of patches along the frequency axis
            num_time_patches (int): Number of patches along the time axis
        """
        super().__init__()
        from transformers import ViTModel
        
        # Load pre-trained Vision Transformer backbone
        local_only = os.getenv("SOUND_DEMO_OFFLINE", "1") != "0"
        self.vit = ViTModel.from_pretrained(model_name, local_files_only=local_only)
        
        # 1. Adapt the patch projection layer to accept 1-channel audio spectrograms
        adapt_projection_layer(self.vit, stride=(10, 10))
        
        # 2. Interpolate the positional embeddings to fit the spectrogram patch sequence length
        interpolate_pos_embeddings(self.vit, num_freq_patches, num_time_patches)
        
        # 3. Add a multi-label classification head
        self.mlp_head = nn.Linear(self.vit.config.hidden_size, num_classes)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Log-Mel Spectrogram input of shape [Batch, 1, Freq, Time]
                              Example: [Batch, 1, 128, 998]
        Returns:
            torch.Tensor: Logits for each class of shape [Batch, num_classes]
        """
        # Run backbone forward pass
        outputs = self.vit(x)
        
        # Extract the representation of the [CLS] classification token (first token)
        cls_token_output = outputs.last_hidden_state[:, 0, :]  # Shape: [Batch, hidden_size]
        
        # Map to final logits
        logits = self.mlp_head(cls_token_output)  # Shape: [Batch, num_classes]
        return logits
