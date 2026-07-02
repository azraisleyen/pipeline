#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config_loader import ConfigLoader
from src.online.api_client import OnlineApiClient
from src.online.online_runner import OnlineRunner
from src.pipeline.orchestrator import PipelineOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TEKNOFEST online simulation loop")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--base-url", default=None, help="Override online.base_url")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--allow-missing-models", action="store_true", help="Schema/client dry-run mode without external weights")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    cfg = ConfigLoader(args.config_dir).load_all()
    if args.base_url is not None:
        cfg.setdefault("online", {})["base_url"] = args.base_url
    if args.allow_missing_models:
        cfg.setdefault("task1", {})["allow_missing_models"] = True
        cfg.setdefault("task3", {})["allow_unavailable"] = True
        cfg.setdefault("pipeline", {})["startup_validation"] = False

    orchestrator = PipelineOrchestrator(cfg)
    runner = OnlineRunner(orchestrator, OnlineApiClient.from_config(cfg), fail_fast=args.fail_fast)
    responses = runner.run(max_frames=args.max_frames)
    print(f"Submitted {len(responses)} packet(s). Health: {orchestrator.health.status()}")


if __name__ == "__main__":
    main()
