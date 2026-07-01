class HealthMonitor:
    def __init__(self): self.errors=[]
    def record_error(self, exc): self.errors.append(str(exc))
    def status(self): return {'ok': not self.errors, 'errors': list(self.errors)}
