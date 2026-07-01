from .task3_state import Task3State
from .v501_adapter import Task3V501Adapter
class Task3Module:
    def __init__(self, config): self.config=config; self.state=Task3State(); self.adapter=Task3V501Adapter(config)
    def initialize(self, session_info=None): self.adapter.initialize(session_info); self.state.initialized=True
    def process(self, context):
        if not self.state.initialized and not self.config.get('task3',{}).get('allow_unavailable',False): self.initialize()
        self.state.frame_count+=1; return self.adapter.process(context)
    def reset(self): self.adapter.reset(); self.state.reset()
