from pathlib import Path
from .exceptions import ModelPathError
class ModelRegistry:
    def __init__(self, config, repo_root=None): self.config=config; self.repo_root=Path(repo_root or config.get('repo_root','.' ))
    def resolve(self, key, required=False, allow_archive=False):
        cur=self.config.get('model_paths',{})
        for part in key.split('.'):
            cur=cur.get(part) if isinstance(cur,dict) else None
        if not cur: 
            if required: raise ModelPathError(f'Model path not configured: {key}')
            return None
        p=Path(cur); p=p if p.is_absolute() else self.repo_root/p
        if p.suffix.lower()=='.zip' and not allow_archive: raise ModelPathError(f'Archive is not a usable model weight: {p}; configure extracted .pt/.pth path')
        if required and not p.exists(): raise ModelPathError(f'Required external model file is missing: {p}')
        return p
