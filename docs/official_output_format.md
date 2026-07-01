# Official Output Format

Every frame produces one JSON packet:

```json
{
  "id": "prediction_000001",
  "user": "",
  "frame": "frame_000001",
  "detected_objects": [],
  "detected_translations": [{"translation_x": 0.0, "translation_y": 0.0, "translation_z": 0.0}],
  "detected_undefined_objects": []
}
```

`detected_translations` is required and must never be empty.
