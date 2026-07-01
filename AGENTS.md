# Codex Instructions

This repository is for the TEKNOFEST 2026 Aviation AI competition.

## Main objective

Build a modular frame-by-frame inference pipeline that will eventually combine:

- Task 1: object detection, landing suitability, and vehicle motion status
- Task 2: position estimation
- Task 3: dynamic reference object matching

The currently included code is the Task 3 v5.0.1 system located under:

```text
third_party/dynamic_reference_system_v5_0_1/
```

## Task 3 integration rule

Do not make `run.py` or `VideoRunner` the main online pipeline entrypoint.

The intended integration approach is:

- keep Task 3 v5.0.1 isolated under `third_party/`
- create a wrapper/adapter in `src/task3_reference/`
- initialize references once per session
- keep tracking state across frames
- process exactly one frame at a time
- return only official-format `detected_undefined_objects`

## Expected Task 3 output

Task 3 adapter should return a list like:

```json
[
  {
    "object_id": "ref_01",
    "top_left_x": 100,
    "top_left_y": 120,
    "bottom_right_x": 250,
    "bottom_right_y": 300
  }
]
```

The main packet builder will place this list under:

```json
"detected_undefined_objects": []
```

## Output contract for final pipeline

Every processed frame must produce one JSON packet with:

```json
{
  "id": "...",
  "user": "...",
  "frame": "...",
  "detected_objects": [],
  "detected_translations": [],
  "detected_undefined_objects": []
}
```

Never omit `detected_translations`. Until the real Task 2 model is integrated, use a stub module.

## Engineering rules

- Do not commit model weights.
- Do not commit datasets, videos, outputs, cache, logs, or debug files.
- Do not hard-code Google Drive absolute paths in reusable modules.
- Use config files for paths and thresholds.
- Keep Task 1, Task 2, and Task 3 independent.
- The pipeline orchestrator is the only component that combines task outputs.
- Add tests for output schema compliance when implementing the main pipeline.
