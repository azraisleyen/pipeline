#!/usr/bin/env python
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.common.config_loader import ConfigLoader
from src.pipeline.orchestrator import PipelineOrchestrator
from src.online.api_client import OnlineApiClient
from src.online.online_runner import OnlineRunner
def main():
    cfg=ConfigLoader().load_all(); runner=OnlineRunner(PipelineOrchestrator(cfg), OnlineApiClient(cfg.get('online',{}).get('base_url',''))); print('Online runner initialized; bind competition API client to fetch/submit frames.')
if __name__=='__main__': main()
