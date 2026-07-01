from src.task1_detection.postprocess import make_object
def test_task1_object_contract():
 o=make_object('0',[1.2,2,3,4],'-1','1'); assert isinstance(o['top_left_x'],int) and o['cls']=='0' and o['motion_status']=='1'
