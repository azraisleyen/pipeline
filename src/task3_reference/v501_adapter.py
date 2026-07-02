from pathlib import Path
import sys
from .task3_postprocess import internal_frame_result_to_official
class Task3V501Adapter:
    def __init__(self, config):
        self.config=config; self.engine=None; self.base_dir=Path(config.get('task3',{}).get('base_dir','third_party/dynamic_reference_system_v5_0_1'))
    def initialize(self, session_info=None):
        t=self.config.get('task3',{})
        if not t.get('enabled',True): return
        if t.get('allow_unavailable',False): return
        from third_party.dynamic_reference_system_v5_0_1.src.core.config import Config as V501Config
        from third_party.dynamic_reference_system_v5_0_1.src.core.device import get_device
        from third_party.dynamic_reference_system_v5_0_1.src.pipeline.inference_engine import InferenceEngine
        cfg_path=Path(t.get('config_path',self.base_dir/'configs/default.yaml')); cfg_path=cfg_path if cfg_path.is_absolute() else Path.cwd()/cfg_path
        cfg=V501Config(cfg_path); reference_dir=Path((session_info or {}).get('reference_dir') or t.get('reference_dir') or cfg.paths.get('reference_dir','data/references'))
        if not reference_dir.is_absolute(): reference_dir=Path.cwd()/reference_dir
        model_path=Path(t.get('detector_model_path') or self.config.get('model_paths',{}).get('task3',{}).get('detector') or cfg.detector.get('model_path','models/yolov8n.pt')); model_path=model_path if model_path.is_absolute() else Path.cwd()/model_path
        if model_path.suffix.lower()=='.zip': raise ValueError(f'Task3 detector model must be extracted weight, not archive: {model_path}')
        if not model_path.exists(): raise FileNotFoundError(f'Task3 detector model file is missing: {model_path}')
        if not reference_dir.exists(): raise FileNotFoundError(f'Task3 reference directory is missing: {reference_dir}')
        device=get_device(t.get('device') or cfg.runtime.get('device','auto'))
        self.engine=InferenceEngine(model_path=model_path, reference_dir=reference_dir, device=device, config=cfg, base_dir=self.base_dir)
    def process(self, context):
        if self.engine is None:
            if self.config.get('task3',{}).get('allow_unavailable',False): return []
            self.initialize()
        result=self.engine.process_frame(context.frame, context.resolved_frame_name(), context.frame_id)
        return internal_frame_result_to_official(result.get('frame_result',{}))
    def reset(self): self.engine=None
