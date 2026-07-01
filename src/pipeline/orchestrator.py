from src.common.packet_builder import PacketBuilder
from src.common.schema_validation import SchemaValidation
from src.task1_detection.task1_module import Task1Module
from src.task2_position.task2_module import Task2Module
from src.task3_reference.task3_module import Task3Module
from .runtime_state import RuntimeState
class PipelineOrchestrator:
    def __init__(self, config):
        self.config=config; self.state=RuntimeState(); self.packet_builder=PacketBuilder(); self.task1=Task1Module(config); self.task2=Task2Module(config); self.task3=Task3Module(config)
    def initialize_session(self, session_info=None):
        self.task1.initialize(); self.task2.initialize();
        if not self.config.get('task3',{}).get('allow_unavailable',False): self.task3.initialize(session_info)
        self.state.initialized=True
    def process_frame(self, context):
        if not self.state.initialized: self.initialize_session(context.metadata.get('session_info') if hasattr(context,'metadata') else None)
        objects=self.task1.process(context); translations=self.task2.process(context); undefined=self.task3.process(context)
        packet=self.packet_builder.build(context,objects,translations,undefined); SchemaValidation.validate(packet); self.state.processed_frames+=1; return packet
    def finalize_session(self): self.state.initialized=False
    def reset(self): self.task1.reset(); self.task2.reset(); self.task3.reset(); self.state=RuntimeState()
