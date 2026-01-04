import torch
import numpy as np
from PIL import Image


class Inferer:
    def __init__(self, modelClass, model_name, device='cuda', **model_kwargs):
        print(model_kwargs)
        model = modelClass(**model_kwargs)
        model.load_state_dict(torch.load('./models/' + model_name))
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        
        print(f"Inferer initialized on {self.device}")
    
    def predict(self, mask_np):
        mask = np.array(mask_np, dtype=np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(mask)
        
        #TODO: check if this can be made more generic
        if isinstance(output, tuple):
            intensity = output[0].cpu().squeeze().numpy()
            resist = output[1].cpu().squeeze().numpy()
            return intensity, resist
        else:
            return output.cpu().squeeze().numpy()

    def predict_batch(self, masks_np):
        if isinstance(masks_np, list):
            masks_np = np.stack(masks_np)
        
        masks = torch.from_numpy(masks_np).float().unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            output = self.model(masks)
        
        if isinstance(output, tuple):
            intensities = output[0].cpu().squeeze(1).numpy()
            resists = output[1].cpu().squeeze(1).numpy()
            return [(intensities[i], resists[i]) for i in range(len(intensities))]
        else:
            return output.cpu().squeeze(1).numpy()