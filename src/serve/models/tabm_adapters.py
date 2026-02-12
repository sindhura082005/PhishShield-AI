import torch
from .tabnet_backbone import SimpleTabNet

class RefTABMNet:
    def __init__(self, input_dim: int, model_path: str = None, device: str = "cpu"):
        self.device = device
        self.model = SimpleTabNet(input_dim)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def predict_proba(self, X):
        import numpy as np
        x = torch.tensor(X, dtype=torch.float32, device=self.device)
        logits = self.model(x)
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(float)
        return probs
