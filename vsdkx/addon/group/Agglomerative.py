from sklearn.cluster import AgglomerativeClustering
from vsdkx.addon.group.DBSCAN import GroupProcessor


class AgglomerativeGroupDetector(GroupProcessor):
    """
    Clusters the detected bounding boxes into groups, based on the distance
    between the bounding boxes:
    1. Clusters bounding boxes
    2. Separates clusters into groups
    3. Creates one bounding box per group

    Attributes:
        distance_threshold (int | float): Distance threshold required by
        the clustering algorithm
    """

    def __init__(self, addon_config: dict, model_settings: dict,
                 model_config: dict, drawing_config: dict):
        super().__init__(addon_config, model_settings, model_config,
                         drawing_config)
        self.distance_threshold = 0.2

    def agglomerative_clustering(self):
        """
        Inits AgglomerativeClustering & Updates distance threshold

        Returns:
            cluster (AgglomerativeClustering): Clustering object
        """
        cluster = AgglomerativeClustering(
            affinity='euclidean',
            linkage='ward',
            compute_distances=False,
            distance_threshold=self.distance_threshold,
            n_clusters=None)

        return cluster

    def _clustering(self):
        """
        Wrapper method for clustering algorithm

        Returns:
            cluster (AgglomerativeClustering): Clustering object
        """
        return self.agglomerative_clustering()
