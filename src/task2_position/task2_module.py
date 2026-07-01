from .position_state import PositionState
from .health_policy import TranslationHealthPolicy
class Task2Module:
    def __init__(self, config):
        t=config.get('task2',{}); d=t.get('default_translation',{})
        self.default={'translation_x':float(d.get('translation_x',0.0)),'translation_y':float(d.get('translation_y',0.0)),'translation_z':float(d.get('translation_z',0.0))}
        self.state=PositionState(); self.policy=TranslationHealthPolicy(t.get('use_upstream_when_valid',True),t.get('keep_last_valid',True))
    def initialize(self): pass
    def process(self, context):
        tr=self.policy.choose(context,self.state,self.default); self.state.update(tr); return [tr]
    def reset(self): self.state.reset()
