from src.common.frame_context import FrameContext
class OnlineRunner:
    def __init__(self, orchestrator, api_client): self.orchestrator=orchestrator; self.api_client=api_client
    def process_request(self, request): return self.orchestrator.process_frame(FrameContext(frame=request.frame,frame_id=request.frame_id,user=request.user))
