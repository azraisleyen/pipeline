import math
from .constants import *
from .exceptions import SchemaValidationError
class SchemaValidation:
    @staticmethod
    def validate(packet):
        for k in [FIELD_ID,FIELD_USER,FIELD_FRAME,FIELD_OBJECTS,FIELD_TRANSLATIONS,FIELD_UNDEFINED]:
            if k not in packet: raise SchemaValidationError(f'Missing field: {k}')
        if not isinstance(packet[FIELD_TRANSLATIONS],list) or len(packet[FIELD_TRANSLATIONS])<1: raise SchemaValidationError('detected_translations must be non-empty')
        for obj in packet[FIELD_OBJECTS]:
            if str(obj.get('cls')) not in VALID_CLASSES: raise SchemaValidationError('Invalid cls')
            if str(obj.get('landing_status')) not in VALID_LANDING: raise SchemaValidationError('Invalid landing_status')
            if str(obj.get('motion_status')) not in VALID_MOTION: raise SchemaValidationError('Invalid motion_status')
            SchemaValidation._bbox(obj)
        for tr in packet[FIELD_TRANSLATIONS]:
            for k in ['translation_x','translation_y','translation_z']:
                v=tr.get(k); 
                if not isinstance(v,(int,float)) or not math.isfinite(float(v)): raise SchemaValidationError(f'Invalid {k}')
        for obj in packet[FIELD_UNDEFINED]:
            if not isinstance(obj.get('object_id'),str) or not obj.get('object_id'): raise SchemaValidationError('Invalid object_id')
            SchemaValidation._bbox(obj)
        return True
    @staticmethod
    def _bbox(obj):
        for k in ['top_left_x','top_left_y','bottom_right_x','bottom_right_y']:
            if not isinstance(obj.get(k),int): raise SchemaValidationError(f'BBox coordinate must be int: {k}')
