"""
Text Embedder module.
Converts textual responses into numerical vector representations.
Supports:
- Sentence-Transformers (e.g., all-MiniLM-L6-v2)
- Hugging Face transformers embeddings
- Resilient Scikit-Learn TF-IDF vectorizer fallback
"""

import numpy as np
from typing import List, Union


class TextEmbedder:
    """
    Computes dense semantic embeddings for text sequences.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._st_model = None
        self._use_fallback = False

    def _load_model(self):
        if self._st_model is None and not self._use_fallback:
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer(self.model_name)
            except Exception as e:
                print(f"[Info] sentence-transformers not available or download skipped ({e}). Using Scikit-Learn TF-IDF Embedder.")
                self._use_fallback = True

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of text strings into an [N x D] normalized embedding matrix.
        """
        if not texts:
            return np.empty((0, 384))

        self._load_model()

        if not self._use_fallback and self._st_model is not None:
            try:
                embeddings = self._st_model.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                return np.asarray(embeddings, dtype=np.float32)
            except Exception as e:
                print(f"[Warning] SentenceTransformer encoding error: {e}. Switching to TF-IDF fallback.")
                self._use_fallback = True

        # Fallback using TF-IDF + SVD / normalization for dense vectors
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        vectorizer = TfidfVectorizer(ngram_range=(1, 3), lowercase=True, max_features=384)
        sparse_mat = vectorizer.fit_transform(texts)
        dense_mat = sparse_mat.toarray()
        
        # Ensure non-zero shape and normalize L2
        if dense_mat.shape[1] == 0:
            dense_mat = np.random.randn(len(texts), 384)
            
        normalized_mat = normalize(dense_mat, norm='l2', axis=1)
        return np.asarray(normalized_mat, dtype=np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """
        Embed a single text string.
        """
        return self.embed([text])[0]
