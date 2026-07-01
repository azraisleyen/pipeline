from pathlib import Path
class LandingClassifier:
    def __init__(self, weights_path:Path, device='cpu', positive_index=1, flip=False):
        import torch
        from torchvision import models, transforms
        self.torch=torch; self.device=torch.device('cuda' if device!='cpu' and torch.cuda.is_available() else 'cpu'); self.positive_index=positive_index; self.flip=flip
        model=models.resnet50(weights=None); model.fc=torch.nn.Linear(model.fc.in_features,2)
        sd=torch.load(weights_path,map_location=self.device); sd=sd.get('state_dict',sd) if isinstance(sd,dict) else sd; sd={k.removeprefix('module.'):v for k,v in sd.items()}
        model.load_state_dict(sd,strict=True); model.to(self.device).eval(); self.model=model
        self.tf=transforms.Compose([transforms.ToPILImage(),transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    def classify(self,crop):
        if crop is None or crop.size==0: return '0'
        with self.torch.no_grad():
            pred=int(self.model(self.tf(crop).unsqueeze(0).to(self.device)).argmax(1).item())
        suitable = (pred==self.positive_index)
        if self.flip: suitable=not suitable
        return '1' if suitable else '0'
class NullLandingClassifier:
    def classify(self,crop): return '0'
