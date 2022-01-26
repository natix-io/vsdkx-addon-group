from sklearn.cluster import DBSCAN

from vsdkx.addon.group.interfaces import BaseGroupProcessor


class DBSCANGroupProcessor(BaseGroupProcessor):
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

        self.distance_threshold = addon_config.get("distance_threshold", 0.5)

    def dbscan_clustering(self):
        """
        Inits DBSCANClustering & Updates distance threshold

        Returns:
            cluster (DBSCANClustering): Clustering object
        """

        cluster = DBSCAN(eps=self.distance_threshold,
                         min_samples=1,
                         metric='euclidean')

        return cluster

    def _clustering(self):
        """
        Wrapper method for clustering algorithm

        Returns:
            cluster (DBSCANClustering): Clustering object
        """
        return self.dbscan_clustering()
