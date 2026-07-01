# Official Output Format Notes

The final online pipeline should produce one JSON packet per frame.

## Full packet

```json
{
  "id": "prediction_000001",
  "user": "http://localhost/users/4/",
  "frame": "http://localhost/frames/4000/",
  "detected_objects": [],
  "detected_translations": [],
  "detected_undefined_objects": []
}
```

## Task 1: detected_objects

```json
{
  "cls": "0",
  "landing_status": "-1",
  "motion_status": "1",
  "top_left_x": 120,
  "top_left_y": 80,
  "bottom_right_x": 250,
  "bottom_right_y": 190
}
```

Class IDs:

| Object  | cls |
|---------|-----|
| vehicle | "0" |
| human   | "1" |
| UAP     | "2" |
| UAI     | "3" |

Status fields:

| Field                 | Meaning                    |
|----------------------|----------------------------|
| landing_status = "-1" | not a landing area        |
| landing_status = "0"  | not suitable for landing  |
| landing_status = "1"  | suitable for landing      |
| motion_status = "-1"  | motion status not required|
| motion_status = "0"   | stationary vehicle        |
| motion_status = "1"   | moving vehicle            |

## Task 2: detected_translations

```json
{
  "translation_x": 0.0,
  "translation_y": 0.0,
  "translation_z": 0.0
}
```

Until the real Task 2 model is integrated, a stub should still return this field.

## Task 3: detected_undefined_objects

```json
{
  "object_id": "ref_01",
  "top_left_x": 300,
  "top_left_y": 180,
  "bottom_right_x": 430,
  "bottom_right_y": 330
}
```
