# Task 3 v5.0.1 Integration Notes

The v5.0.1 system is preserved under `third_party/dynamic_reference_system_v5_0_1/`. The adapter uses `src.pipeline.inference_engine.InferenceEngine.process_frame(frame_bgr, frame_name, frame_index)` as the frame-by-frame entrypoint, preserving reference-bank initialization and tracking state inside the engine for the full session.

The adapter strips internal scores, candidate details, and debug fields before returning official `detected_undefined_objects` records.
