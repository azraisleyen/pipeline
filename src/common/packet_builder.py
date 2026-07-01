from .constants import FIELD_ID,FIELD_USER,FIELD_FRAME,FIELD_OBJECTS,FIELD_TRANSLATIONS,FIELD_UNDEFINED
class PacketBuilder:
    def build(self, context, detected_objects, detected_translations, detected_undefined_objects):
        return {FIELD_ID:context.resolved_prediction_id(),FIELD_USER:context.user,FIELD_FRAME:context.resolved_frame_name(),FIELD_OBJECTS:list(detected_objects or []),FIELD_TRANSLATIONS:list(detected_translations or []),FIELD_UNDEFINED:list(detected_undefined_objects or [])}
