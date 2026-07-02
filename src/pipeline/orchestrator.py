from __future__ import annotations

from src.common.packet_builder import PacketBuilder
from src.common.schema_validation import SchemaValidation
from src.task1_detection.task1_module import Task1Module
from src.task2_position.task2_module import Task2Module
from src.task3_reference.task3_module import Task3Module

from .health_monitor import HealthMonitor
from .runtime_state import RuntimeState
from .startup_validation import StartupValidator


class PipelineOrchestrator:
    def __init__(self, config, *, validate_startup: bool | None = None, health_monitor: HealthMonitor | None = None):
        self.config = config
        self.state = RuntimeState()
        self.health = health_monitor or HealthMonitor()
        self.packet_builder = PacketBuilder()
        self.task1 = Task1Module(config)
        self.task2 = Task2Module(config)
        self.task3 = Task3Module(config)
        self.validate_startup = config.get("pipeline", {}).get("startup_validation", False) if validate_startup is None else validate_startup
        if self.validate_startup:
            try:
                StartupValidator(config).validate()
            except Exception as exc:
                self.health.record_error(exc, component="startup", recoverable=False)
                raise

    def initialize_session(self, session_info=None):
        self.task1.initialize()
        self.task2.initialize()
        if not self.config.get("task3", {}).get("allow_unavailable", False):
            self.task3.initialize(session_info)
        self.state.initialized = True

    def process_frame(self, context):
        if not self.state.initialized:
            self.initialize_session(context.session_info() if hasattr(context, "session_info") else context.metadata.get("session_info", {}))
        try:
            objects = self.task1.process(context)
            translations = self.task2.process(context)
            undefined = self.task3.process(context)
            packet = self.packet_builder.build(context, objects, translations, undefined)
            if self.config.get("pipeline", {}).get("validate_schema", True):
                SchemaValidation.validate(packet)
            self.state.processed_frames += 1
            return packet
        except Exception as exc:
            self.health.record_error(exc, component="process_frame", recoverable=False, frame_id=getattr(context, "frame_id", None))
            raise

    def finalize_session(self):
        self.task3.reset()
        self.state.initialized = False

    def reset(self):
        self.task1.reset()
        self.task2.reset()
        self.task3.reset()
        self.state = RuntimeState()
