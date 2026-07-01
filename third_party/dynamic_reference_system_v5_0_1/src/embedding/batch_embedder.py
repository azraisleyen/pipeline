
class BatchEmbedder:
    def __init__(self, embedder):
        self.embedder = embedder

    def embed_many(self, images):
        if hasattr(self.embedder, "embed_batch"):
            return self.embedder.embed_batch(images)

        embeddings = []
        for img in images:
            embeddings.append(self.embedder.embed(img))
        return embeddings
