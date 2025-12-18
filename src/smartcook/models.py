import torch
import torch.nn as nn

class MaskedCookingAutoencoder(nn.Module):
    """
    The Core AI Brain.
    Learns to understand cooking physics by filling in missing (masked) data.
    """
    def __init__(self, input_dim=3, hidden_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Encoder: Reads the sensor data and compresses it into a 'hidden' thought
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Decoder: Takes the 'hidden' thought and tries to recreate the sensor data
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x, mask_ratio=0.0):
        # x shape: [batch_size, 60, 3]
        
        # 1. Masking Logic (Self-Supervised Learning)
        # During training, we randomly zero-out some data to make the task harder
        if self.training and mask_ratio > 0:
            mask = torch.rand_like(x) > mask_ratio
            x_input = x * mask.float()
        else:
            x_input = x
            
        # 2. Encode
        # 'hidden' is the context vector (the brain's understanding)
        _, (hidden, _) = self.encoder(x_input)
        
        # 3. Decode
        # We repeat the hidden thought 60 times so the decoder can reconstruct each minute
        repeat_hidden = hidden.permute(1, 0, 2).repeat(1, x.shape[1], 1)
        reconstructed, _ = self.decoder(repeat_hidden)
        
        return reconstructed, hidden


class CookingPredictor(nn.Module):
    """
    The Downstream Task Solver.
    Uses the pretrained brain to answer specific questions:
    - What stage is the food in? (Classification)
    - How much time is left? (Regression)
    """
    def __init__(self, pretrained_model):
        super().__init__()
        # Steal the encoder from the pretrained model
        self.encoder = pretrained_model.encoder
        self.hidden_dim = pretrained_model.hidden_dim
        
        # FREEZE the encoder (Transfer Learning)
        # We don't want to retrain the brain, just the new output layers
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Task 1 Head: Stage Classification (3 classes: Raw, Cooking, Done)
        self.stage_head = nn.Linear(self.hidden_dim, 3)
        
        # Task 2 Head: Time Prediction (1 number: Minutes remaining)
        self.time_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, x):
        # We just want the 'hidden' understanding
        _, (hidden, _) = self.encoder(x)
        embedding = hidden.squeeze(0)
        
        # Ask the heads to make predictions based on that understanding
        stage_logits = self.stage_head(embedding)
        time_pred = self.time_head(embedding)
        
        return stage_logits, time_pred