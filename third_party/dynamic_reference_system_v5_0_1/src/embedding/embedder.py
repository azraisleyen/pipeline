
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel


class DINOEmbedder:
    def __init__(
        self,
        model_name="facebook/dinov2-small",
        device="cpu",
        input_size=224,
        use_fp16=True,
        normalize_embeddings=True,
        pooling="cls",
    ):
        self.model_name = model_name
        self.device = device
        self.input_size = int(input_size)
        self.use_fp16 = bool(use_fp16)
        self.normalize_embeddings = bool(normalize_embeddings)
        self.pooling = str(pooling).lower()

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

        if self.use_fp16 and self.device == "cuda":
            self.model = self.model.half()

    def _prepare_inputs(self, image_bgr):
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Embedder'a boş görüntü verildi.")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        image_rgb = cv2.resize(
            image_rgb,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_AREA,
        )

        inputs = self.processor(images=image_rgb, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].to(self.device)

            if (
                self.use_fp16
                and self.device == "cuda"
                and inputs[key].dtype == torch.float32
            ):
                inputs[key] = inputs[key].half()

        return inputs

    def embed(self, image_bgr):
        inputs = self._prepare_inputs(image_bgr)

        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state

            if self.pooling == "cls":
                feat = hidden[:, 0]
            elif self.pooling == "mean":
                feat = hidden.mean(dim=1)
            else:
                raise ValueError(f"Geçersiz pooling türü: {self.pooling}")

        feat = feat.float().cpu().numpy()[0].astype(np.float32)

        if self.normalize_embeddings:
            norm = np.linalg.norm(feat)
            if norm > 1e-8:
                feat = feat / norm

        return feat

    @staticmethod
    def cosine(vec1, vec2) -> float:
        vec1 = np.asarray(vec1, dtype=np.float32)
        vec2 = np.asarray(vec2, dtype=np.float32)

        denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2)) + 1e-8
        score = float(np.dot(vec1, vec2) / denom)

        if np.isnan(score):
            return 0.0

        return score
