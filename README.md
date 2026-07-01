# TEKNOFEST 2026 Aviation AI Pipeline

Production-ready modular pipeline for frame-by-frame TEKNOFEST online simulation packets.

## Runtime flow

`FrameContext -> Task1Module -> Task2Module -> Task3Module -> PacketBuilder -> SchemaValidation -> official JSON packet`

## External models

Weights are intentionally not committed. Configure paths in `configs/model_paths.yaml` and place weights under:

- `models/task1/elcey.pt`
- `models/task1/vehicle.pt`
- `models/task1/UAP_UAI_V2.pt`
- `models/task1/UAP_UAI_Classifier_resnet50_V4.1.pth`

Task 3 remains isolated under `third_party/dynamic_reference_system_v5_0_1/` and is integrated through `src/task3_reference/v501_adapter.py`.

## Run

```bash
python scripts/run_single_frame.py path/to/frame.jpg
python scripts/run_offline_dataset.py path/to/images
python scripts/run_offline_video.py path/to/video.mp4 --max-frames 100
python scripts/run_online_simulation.py
```

For schema-only smoke runs without external weights, pass `--allow-missing-models` to local scripts.

## Test

```bash
pytest
```
