from src.task3_reference.task3_postprocess import internal_frame_result_to_official
def test_task3_postprocess_contract():
 out=internal_frame_result_to_official({'objects':[{'reference_id':'ref_01','bbox':[1,2,3,4],'score':.9}]}); assert out==[{'object_id':'ref_01','top_left_x':1,'top_left_y':2,'bottom_right_x':3,'bottom_right_y':4}]
