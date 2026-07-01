# TEKNOFEST 2026 Aviation AI Pipeline

This repository is prepared for building a modular TEKNOFEST 2026 Aviation AI pipeline.

The final target pipeline will combine:

1. Task 1 — Object detection, landing suitability, and vehicle motion status
2. Task 2 — Position estimation
3. Task 3 — Dynamic reference object matching

At this stage, the repository includes the cleaned Task 3 v5.0.1 reference matching system under:

```text
third_party/dynamic_reference_system_v5_0_1/
```

## Current purpose

This repository is intentionally prepared as a clean GitHub-ready base for Codex review and later integration.

The Task 3 system should be integrated into the main pipeline through an adapter layer, not by directly using the standalone `run.py` or offline `VideoRunner` as the main runtime entry.

## Planned integration

The future main pipeline should use this structure:

```text
src/
  common/
  task1_detection/
  task2_position/
  task3_reference/
  pipeline/
  online/
  evaluation/
```

Task 3 should expose only:

```python
detected_undefined_objects = task3_module.process(frame_context)
```

The main packet builder will combine:

```json
{
  "detected_objects": [],
  "detected_translations": [],
  "detected_undefined_objects": []
}
```

## Important repository rules

Model weights, videos, datasets, cache files, debug outputs, and generated result files are intentionally excluded from Git.

Do not commit:

```text
*.pt
*.pth
*.pth.tar
*.onnx
*.mp4
data/
outputs/
logs/
cache/
models/
```

## Next step

After pushing this repository to GitHub, ask Codex to inspect the Task 3 v5.0.1 structure and identify the correct integration point for a `Task3V501Adapter`.
