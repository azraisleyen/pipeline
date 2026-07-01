from pathlib import Path
class YoloDetector:
    def __init__(self, model_path:Path, conf:float, iou:float, imgsz:int, max_det:int, device='auto'):
        from ultralytics import YOLO
        self.model=YOLO(str(model_path)); self.conf=conf; self.iou=iou; self.imgsz=imgsz; self.max_det=max_det; self.device=device
    def predict(self, frame):
        results=self.model.predict(source=frame, device=None if self.device=='auto' else self.device, imgsz=self.imgsz, conf=self.conf, iou=self.iou, max_det=self.max_det, verbose=False)
        out=[]
        for r in results:
            boxes=getattr(r,'boxes',None)
            if boxes is None: continue
            for b in boxes:
                xyxy=b.xyxy[0].detach().cpu().tolist(); conf=float(b.conf[0]) if b.conf is not None else 0.0; cls=int(b.cls[0]) if b.cls is not None else 0
                out.append({'bbox':xyxy,'score':conf,'model_cls':cls})
        return out
class NullDetector:
    def predict(self, frame): return []
