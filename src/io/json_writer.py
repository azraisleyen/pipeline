import json
from pathlib import Path
class JsonWriter:
    def __init__(self,path): self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True); self.items=[]
    def write_packet(self,packet): self.items.append(packet); self.path.write_text(json.dumps(self.items,ensure_ascii=False,indent=2),encoding='utf-8')
