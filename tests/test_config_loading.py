from src.common.config_loader import ConfigLoader
def test_config_loads():
 cfg=ConfigLoader().load_all(); assert 'task1' in cfg and 'task2' in cfg and 'task3' in cfg and 'model_paths' in cfg
