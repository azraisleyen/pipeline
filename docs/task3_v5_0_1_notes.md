# Task 3 v5.0.1 Notes

This document summarizes how the Task 3 dynamic reference object matching system should be treated during main pipeline integration.

## Purpose

Task 3 receives reference object images at session start and detects those objects in incoming frames.

## Integration principle

The v5.0.1 system should remain isolated under:

```text
third_party/dynamic_reference_system_v5_0_1/
```

The main pipeline should not directly depend on standalone scripts.

The desired integration point is a future adapter:

```text
src/task3_reference/v501_adapter.py
```

## Runtime behavior

The adapter should:

1. Load the v5.0.1 config.
2. Initialize the Task 3 inference engine once per session.
3. Build or load the reference bank once per session.
4. Keep tracking state across frames.
5. Process one frame at a time.
6. Return only official-format `detected_undefined_objects`.

## Offline-only components

The following components may be useful for testing but should not be used as the main online runtime:

- `run.py`
- `VideoRunner`
- dataset/video batch runners

## Official Task 3 output

The adapter should return:

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

## Notes for Codex

When implementing the main pipeline, inspect:

- `src/pipeline/inference_engine.py`
- `src/output/official_adapter.py`
- `src/tracking/`
- `src/scoring/`
- `src/matching/`
- `configs/default.yaml`

Then create a thin adapter rather than rewriting the Task 3 internals.
