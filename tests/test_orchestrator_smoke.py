import numpy as np
from src.common.frame_context import FrameContext
from src.pipeline.orchestrator import PipelineOrchestrator
def test_orchestrator_smoke_without_external_models():
 cfg={'task1':{'allow_missing_models':True},'task2':{},'task3':{'allow_unavailable':True}}
 p=PipelineOrchestrator(cfg).process_frame(FrameContext(np.zeros((8,8,3),dtype=np.uint8),1))
 assert p['detected_translations'] and 'detected_objects' in p and 'detected_undefined_objects' in p
