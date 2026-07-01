class Task2StubEngine:
    def __init__(self, default_translation): self.default_translation=default_translation
    def infer(self, context): return dict(self.default_translation)
