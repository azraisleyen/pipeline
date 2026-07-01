# Integration Contracts

- `legacy/prova_V3.py` is reference-only and is not a runtime entrypoint.
- `third_party/dynamic_reference_system_v5_0_1/run.py` and `VideoRunner` remain standalone/offline code and are not online entrypoints.
- Online and offline runners both call `PipelineOrchestrator.process_frame()`.
- Official output packets contain only `id`, `user`, `frame`, `detected_objects`, `detected_translations`, and `detected_undefined_objects`.
