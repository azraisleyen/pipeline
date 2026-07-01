from src.common.bbox_ops import clamp_bbox
def internal_frame_result_to_official(frame_result):
    out=[]
    for obj in (frame_result or {}).get('objects',[]):
        bbox=obj.get('bbox')
        if not bbox or len(bbox)!=4: continue
        x1,y1,x2,y2=clamp_bbox(bbox)
        ref=str(obj.get('reference_id') or obj.get('object_id') or '')
        if ref: out.append({'object_id':ref,'top_left_x':x1,'top_left_y':y1,'bottom_right_x':x2,'bottom_right_y':y2})
    return out
