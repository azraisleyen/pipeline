import json
from pathlib import Path
class ExportRunner:
    def export(self, packets, path): Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text(json.dumps(packets,indent=2),encoding='utf-8')
