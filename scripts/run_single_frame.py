#!/usr/bin/env python
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import argparse, json, cv2
from src.common.config_loader import ConfigLoader
from src.common.frame_context import FrameContext
from src.pipeline.orchestrator import PipelineOrchestrator
p=argparse.ArgumentParser(); p.add_argument('image'); p.add_argument('--config-dir',default='configs'); p.add_argument('--allow-missing-models',action='store_true'); a=p.parse_args()
cfg=ConfigLoader(a.config_dir).load_all()
if a.allow_missing_models: cfg['task1']['allow_missing_models']=True; cfg['task3']['allow_unavailable']=True
frame=cv2.imread(a.image); orch=PipelineOrchestrator(cfg); print(json.dumps(orch.process_frame(FrameContext(frame,1)),ensure_ascii=False,indent=2))
