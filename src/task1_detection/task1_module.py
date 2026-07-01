from src.common.constants import CLASS_HUMAN,CLASS_VEHICLE,CLASS_UAP,CLASS_UAI,LANDING_NOT_APPLICABLE,MOTION_NOT_APPLICABLE
from src.common.image_ops import crop
from src.common.model_registry import ModelRegistry
from .detectors import YoloDetector, NullDetector
from .landing_classifier import LandingClassifier, NullLandingClassifier
from .motion_estimator import VehicleMotionEstimator
from .postprocess import make_object,nms_objects
class Task1Module:
    def __init__(self, config, lazy=True):
        self.config=config; t=config.get('task1',{}); self.enabled=t.get('enabled',True); self.lazy=lazy; self.loaded=False
        self.motion=VehicleMotionEstimator(t.get('motion_threshold_px',12.0), t.get('motion_max_lost_frames',5))
    def initialize(self):
        if self.loaded: return
        t=self.config.get('task1',{}); reg=ModelRegistry(self.config)
        if not self.enabled or t.get('allow_missing_models',True): self.human=self.vehicle=self.landing=NullDetector(); self.classifier=NullLandingClassifier(); self.loaded=True; return
        self.human=YoloDetector(reg.resolve('task1.human',True),t.get('human_conf',0.25),t.get('iou',0.65),t.get('imgsz',1280),t.get('max_det',500),t.get('device','auto'))
        self.vehicle=YoloDetector(reg.resolve('task1.vehicle',True),t.get('vehicle_conf',0.4),t.get('iou',0.65),t.get('imgsz',1280),t.get('max_det',500),t.get('device','auto'))
        self.landing=YoloDetector(reg.resolve('task1.uap_uai',True),t.get('landing_conf',0.4),t.get('iou',0.65),t.get('imgsz',1280),t.get('max_det',500),t.get('device','auto'))
        self.classifier=LandingClassifier(reg.resolve('task1.landing_classifier',True),t.get('device','cpu'),t.get('classifier_positive_index',1),t.get('flip_landing_classes',False)); self.loaded=True
    def process(self, context):
        if not self.loaded: self.initialize()
        if not self.enabled: return []
        objs=[]
        for d in self.human.predict(context.frame): objs.append(make_object(CLASS_HUMAN,d['bbox']))
        for d in self.vehicle.predict(context.frame): objs.append(make_object(CLASS_VEHICLE,d['bbox'],LANDING_NOT_APPLICABLE,self.motion.estimate(d['bbox'],context.frame_id)))
        for d in self.landing.predict(context.frame):
            cls=CLASS_UAP if int(d.get('model_cls',0))==0 else CLASS_UAI; bbox=tuple(map(int,d['bbox'])); objs.append(make_object(cls,bbox,self.classifier.classify(crop(context.frame,bbox)),MOTION_NOT_APPLICABLE))
        return nms_objects(objs,self.config.get('task1',{}).get('extra_nms_iou',0.75))
    def reset(self): self.motion.reset(); self.loaded=False
