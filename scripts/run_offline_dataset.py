#!/usr/bin/env python
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import argparse,json
from src.common.config_loader import ConfigLoader
from src.pipeline.orchestrator import PipelineOrchestrator
from src.offline.dataset_runner import OfflineDatasetRunner
p=argparse.ArgumentParser(); p.add_argument('dataset_dir'); p.add_argument('--allow-missing-models',action='store_true'); a=p.parse_args(); cfg=ConfigLoader().load_all()
if a.allow_missing_models: cfg['task1']['allow_missing_models']=True; cfg['task3']['allow_unavailable']=True
print(json.dumps(OfflineDatasetRunner(PipelineOrchestrator(cfg)).run(a.dataset_dir),indent=2))
