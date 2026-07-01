
from pathlib import Path
import json


def _clamp_int(value, low=0, high=None):
    value = int(round(float(value)))
    if high is not None:
        value = min(value, int(high))
    return max(int(low), value)


def convert_internal_to_official(internal_results, object_key="detected_undefined_objects"):
    official_frames = []

    for frame in internal_results.get("frames", []):
        objects = []

        for obj in frame.get("objects", []):
            bbox = obj.get("bbox", None)
            if bbox is None or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox

            objects.append(
                {
                    "object_id": obj.get("reference_id"),
                    "top_left_x": _clamp_int(x1),
                    "top_left_y": _clamp_int(y1),
                    "bottom_right_x": _clamp_int(x2),
                    "bottom_right_y": _clamp_int(y2),
                }
            )

        official_frames.append(
            {
                "frame_name": frame.get("frame_name"),
                object_key: objects,
            }
        )

    return {"frames": official_frames}


def save_official_json(internal_results, output_path, object_key="detected_undefined_objects"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    official = convert_internal_to_official(
        internal_results=internal_results,
        object_key=object_key,
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(official, f, ensure_ascii=False, indent=2)

    return official
