from src.common.bbox_ops import clamp_bbox, iou
from src.common.constants import LANDING_NOT_APPLICABLE, MOTION_NOT_APPLICABLE
def make_object(cls,bbox,landing_status=LANDING_NOT_APPLICABLE,motion_status=MOTION_NOT_APPLICABLE):
    x1,y1,x2,y2=clamp_bbox(bbox)
    return {'cls':str(cls),'landing_status':str(landing_status),'motion_status':str(motion_status),'top_left_x':x1,'top_left_y':y1,'bottom_right_x':x2,'bottom_right_y':y2}
def nms_objects(objects, iou_threshold=0.75):
    kept=[]
    for obj in objects:
        b=(obj['top_left_x'],obj['top_left_y'],obj['bottom_right_x'],obj['bottom_right_y'])
        if all(iou(b,(k['top_left_x'],k['top_left_y'],k['bottom_right_x'],k['bottom_right_y']))<iou_threshold for k in kept): kept.append(obj)
    return kept
