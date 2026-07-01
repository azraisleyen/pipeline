from src.common.schema_validation import SchemaValidation
def test_schema_validates_packet():
 assert SchemaValidation.validate({'id':'p','user':'','frame':'f','detected_objects':[],'detected_translations':[{'translation_x':0.0,'translation_y':0.0,'translation_z':0.0}],'detected_undefined_objects':[]})
