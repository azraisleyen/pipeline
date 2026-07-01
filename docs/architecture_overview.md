# Architecture Overview

The production pipeline processes exactly one frame at a time through `PipelineOrchestrator`:

`FrameContext -> Task1Module -> Task2Module -> Task3Module -> PacketBuilder -> SchemaValidation`.

Task 1 loads YOLO and ResNet weights from config-driven paths. Task 2 always returns a non-empty translation list using upstream data when valid or configured defaults otherwise. Task 3 keeps v5.0.1 isolated under `third_party/` and calls its frame-level `InferenceEngine.process_frame()` through `Task3V501Adapter`.
