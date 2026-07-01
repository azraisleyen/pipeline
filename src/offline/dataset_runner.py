from src.common.frame_context import FrameContext
from src.io.frame_sources import ImageDirectorySource
class OfflineDatasetRunner:
    def __init__(self, orchestrator): self.orchestrator=orchestrator
    def run(self, dataset_dir):
        return [self.orchestrator.process_frame(FrameContext(frame=fr,frame_id=i,frame_name=name)) for i,name,fr in ImageDirectorySource(dataset_dir)]
