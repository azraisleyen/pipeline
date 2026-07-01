from pathlib import Path
class SessionReferenceManager:
    def __init__(self, reference_dir): self.reference_dir=Path(reference_dir) if reference_dir else None
    def resolve(self, session_info=None):
        if session_info and session_info.get('reference_dir'): return Path(session_info['reference_dir'])
        return self.reference_dir
