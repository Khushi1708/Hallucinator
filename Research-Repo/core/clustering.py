"""
Semantic Clustering Module.
Calculates cosine similarity between embeddings and groups semantically equivalent responses
into distinct clusters.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA


class SemanticClusterer:
    """
    Clusters LLM responses based on cosine similarity of dense embeddings.
    """

    def __init__(self, similarity_threshold: float = 0.75):
        """
        :param similarity_threshold: Minimum cosine similarity to merge into same semantic cluster.
        """
        self.similarity_threshold = similarity_threshold

    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate NxN pairwise cosine similarity matrix.
        """
        if embeddings.shape[0] == 0:
            return np.empty((0, 0))
        return cosine_similarity(embeddings)

    def cluster(
        self,
        responses: List[str],
        embeddings: np.ndarray
    ) -> Dict[str, Any]:
        """
        Cluster responses into semantic equivalence classes.
        Returns:
            - cluster_labels: List of cluster IDs per response
            - clusters: Dict mapping cluster_id -> list of response indices
            - cluster_probabilities: Dict mapping cluster_id -> fraction of responses
            - cluster_representatives: Dict mapping cluster_id -> representative text (medoid)
            - similarity_matrix: NxN cosine similarity matrix
            - coords_2d: Nx2 PCA coordinates for scatter plot visualization
        """
        n = len(responses)
        if n == 0:
            return {
                "cluster_labels": [],
                "clusters": {},
                "cluster_probabilities": {},
                "cluster_representatives": {},
                "similarity_matrix": np.empty((0, 0)),
                "coords_2d": np.empty((0, 2)),
            }

        sim_matrix = self.compute_similarity_matrix(embeddings)

        # If only 1 sample, trivial single cluster
        if n == 1:
            return {
                "cluster_labels": [0],
                "clusters": {0: [0]},
                "cluster_probabilities": {0: 1.0},
                "cluster_representatives": {0: responses[0]},
                "similarity_matrix": sim_matrix,
                "coords_2d": np.zeros((1, 2)),
            }

        # Distance matrix for agglomerative clustering: dist = 1 - sim
        # Clip to ensure valid non-negative distance
        distance_matrix = np.clip(1.0 - sim_matrix, 0.0, 2.0)
        distance_threshold = 1.0 - self.similarity_threshold

        try:
            clustering_model = AgglomerativeClustering(
                n_clusters=None,
                metric="precomputed",
                linkage="average",
                distance_threshold=distance_threshold
            )
            labels = clustering_model.fit_predict(distance_matrix).tolist()
        except Exception:
            # Fallback simple threshold connected components
            labels = self._connected_components_clustering(sim_matrix, self.similarity_threshold)

        # Build cluster mapping
        clusters: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(label, []).append(idx)

        # Sort clusters by size descending
        sorted_cluster_ids = sorted(clusters.keys(), key=lambda cid: len(clusters[cid]), reverse=True)
        remapped_clusters: Dict[int, List[int]] = {}
        remapped_labels = [0] * n
        for new_id, old_id in enumerate(sorted_cluster_ids):
            remapped_clusters[new_id] = clusters[old_id]
            for idx in clusters[old_id]:
                remapped_labels[idx] = new_id

        # Compute cluster probabilities
        cluster_probs: Dict[int, float] = {
            cid: len(indices) / n for cid, indices in remapped_clusters.items()
        }

        # Find representative response for each cluster (medoid: response with highest average intra-cluster similarity)
        cluster_reps: Dict[int, str] = {}
        for cid, indices in remapped_clusters.items():
            if len(indices) == 1:
                cluster_reps[cid] = responses[indices[0]]
            else:
                sub_sim = sim_matrix[np.ix_(indices, indices)]
                avg_sims = sub_sim.mean(axis=1)
                medoid_idx = indices[int(np.argmax(avg_sims))]
                cluster_reps[cid] = responses[medoid_idx]

        # 2D projection using PCA for visualization
        if n >= 2:
            n_components = min(2, embeddings.shape[1], n)
            pca = PCA(n_components=n_components)
            coords = pca.fit_transform(embeddings)
            if coords.shape[1] == 1:
                coords = np.column_stack([coords, np.zeros((n, 1))])
        else:
            coords = np.zeros((n, 2))

        return {
            "cluster_labels": remapped_labels,
            "clusters": remapped_clusters,
            "cluster_probabilities": cluster_probs,
            "cluster_representatives": cluster_reps,
            "similarity_matrix": sim_matrix,
            "coords_2d": coords,
        }

    def _connected_components_clustering(self, sim_matrix: np.ndarray, threshold: float) -> List[int]:
        n = sim_matrix.shape[0]
        visited = [False] * n
        labels = [-1] * n
        current_label = 0

        for i in range(n):
            if not visited[i]:
                queue = [i]
                visited[i] = True
                labels[i] = current_label
                while queue:
                    curr = queue.pop(0)
                    for j in range(n):
                        if not visited[j] and sim_matrix[curr, j] >= threshold:
                            visited[j] = True
                            labels[j] = current_label
                            queue.append(j)
                current_label += 1
        return labels
