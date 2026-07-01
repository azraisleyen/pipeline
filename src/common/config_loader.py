from pathlib import Path
import copy, yaml
from .exceptions import ConfigError
def deep_merge(a,b):
    out=copy.deepcopy(a)
    for k,v in (b or {}).items(): out[k]=deep_merge(out[k],v) if isinstance(out.get(k),dict) and isinstance(v,dict) else copy.deepcopy(v)
    return out
class ConfigLoader:
    def __init__(self, config_dir='configs', repo_root=None): self.repo_root=Path(repo_root or Path.cwd()); self.config_dir=(self.repo_root/config_dir)
    def load_yaml(self,name):
        p=self.config_dir/name
        if not p.exists(): raise ConfigError(f'Missing config file: {p}')
        with p.open(encoding='utf-8') as f: return yaml.safe_load(f) or {}
    def load_all(self):
        cfg={'repo_root':str(self.repo_root)}
        for name in ['pipeline.yaml','model_paths.yaml','task1.yaml','task2.yaml','task3_v501.yaml','online.yaml','logging.yaml']:
            if (self.config_dir/name).exists(): cfg=deep_merge(cfg,self.load_yaml(name))
        return cfg
