import numpy as np
import networkx as nx

from scipy.spatial import distance as dist
from sklearn.preprocessing import MinMaxScaler
from chinese_whispers import chinese_whispers, aggregate_clusters

from vsdkx.core.structs import AddonObject, Inference
from vsdkx.addon.group.interfaces import BaseGroupProcessor


class ChineseWhispersGroupProcessor(BaseGroupProcessor):
    """
    Clusters the detected bounding boxes into groups, based on the distance
    between the bounding boxes:
    1. Clusters bounding boxes
    2. Separates clusters into groups
    3. Creates one bounding box per group

    Attributes:
        distance_threshold (int | float): Distance threshold required by
        the clustering algorithm
        min_group_size (int): Minimum amount of detected people to be
        considered a group
        temporal_len (int): Size of temporal data to be used
        feat_size (int): Amount of features used in the algorithm
    """

    def __init__(self, addon_config: dict, model_settings: dict,
                 model_config: dict, drawing_config: dict):
        """
        Args:
            min_group_size (int): Minimum amount of detected people to be
            considered a group
        """
        super().__init__(addon_config, model_settings, model_config,
                         drawing_config)

        self.distance_threshold = addon_config.get("distance_threshold", 0.2)

    def get_features(self, boxes, trackable_objects):
        """
        Calculates each boxes' centroid
        Args:
            boxes (list): List of bounding boxes
            trackable_objects (TrackableObjects): List with trackable objects

        Returns:
            (list): List with centroids
        """

        centroids = []

        pi = 3.14
        degrees_half = 180
        degrees_full = 360
        filtered_boxes = []

        for box in boxes:
            c = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))

            for to_id, to_obj in trackable_objects.items():
                if np.array_equal(c, to_obj.centroids[-1]) \
                        and np.array_equal(box.astype(int),
                                           to_obj.bounding_box.astype(int)):
                    box = to_obj.bounding_box
                    width = int(box[2] - box[0])
                    height = int(box[3] - box[1])

                    distance = (2 * pi * degrees_half) / \
                               (width + height * degrees_full) * 1000 + 3

                    if len(to_obj.centroids) >= self.temporal_len:
                        to_obj_centroids = np.array(
                            to_obj.centroids[-self.temporal_len:])
                        to_obj_centroids = np.array(to_obj_centroids)

                        to_obj_centroids = np.append(to_obj_centroids, distance)

                    else:
                        to_cen_size = len(to_obj.centroids)
                        centroids_flatten = np.array(to_obj.centroids).flatten()
                        zeros_array = np.zeros(self.temporal_len * 2)
                        zeros_array[
                        ((self.temporal_len * 2) - (to_cen_size * 2)):
                        (self.temporal_len * 2)] = \
                            centroids_flatten[0:(to_cen_size * 2)]
                        to_obj_centroids = zeros_array
                        to_obj_centroids = np.append(to_obj_centroids, distance)

                    centroids.append(to_obj_centroids.flatten())
                    filtered_boxes.append(box)

        identity_matrix = []
        # Create indentify matrix with IOUs per bounding box
        for box_a in filtered_boxes:
            iou_row = np.zeros(len(filtered_boxes))
            for i, box_b in enumerate(filtered_boxes):
                iou_row[i] = self.bb_intersection_over_union(box_a, box_b)
            identity_matrix.append(iou_row)

        identity_matrix = np.array(identity_matrix)
        # Create a graph to get the neighborhoods
        G = nx.from_numpy_matrix(identity_matrix)

        id_counter = 100  # Neighborhood ID counter
        n_ids = []
        neighbourhoods = []
        n_feat = []
        for i in range(0, len(centroids)):
            neighbourhood = list(G.neighbors(i))
            if neighbourhood in neighbourhoods:
                idx = neighbourhoods.index(neighbourhood)
                n_id = n_ids[idx]
                n_feat.append(n_id)
            else:
                neighbourhoods.append(neighbourhood)
                id_counter += 50
                n_ids.append(id_counter)
                n_feat.append(id_counter)

        centroids = np.array(centroids)

        n_feat = np.array(n_feat)

        features = np.zeros((len(centroids), self.feat_size))
        features[:, 0: self.temporal_len * 2] = \
            centroids[:, 0:self.temporal_len * 2]
        features[:, self.feat_size - 3] = centroids[:, self.temporal_len * 2]
        features[:, self.feat_size - 2] = n_feat.flatten()

        dx = centroids[:, self.temporal_len * 2 - 2] - centroids[:, 0]
        dy = centroids[:, self.temporal_len * 2 - 1] - centroids[:, 1]
        features[:, self.feat_size - 1] = np.arctan2(dy, dx)

        if len(boxes) < len(features):
            print('ERROR! ')

        features = dist.cdist(features, features, metric='euclidean')
        features = MinMaxScaler().fit_transform(features)

        features[features > self.distance_threshold] = 0

        return features, centroids[:, 4:6], np.array(filtered_boxes)

    def get_cluster_boxes(self, boxes, indexes, centroids):
        """
        Separates boxes by their cluster ID

        Args:
            boxes (np.array): Array with bounding boxes
            indexes (list): List of cluster IDs
            centroids (tuple): Tuple with x,y values of a centroid

        Returns:
            (np.array): Array with bounding boxes
            (np.array): Array with centroid x,y points
        """

        cluster_boxes = []
        centroids_list = []
        for idx in indexes:
            box = boxes[idx]
            centroid = centroids[idx]
            cluster_boxes.append(box)
            centroids_list.append(centroid)
        return np.array(cluster_boxes), np.array(centroids_list)

    def _clustering(self, features):
        graph = nx.from_numpy_matrix(features)
        chinese_whispers(graph, seed=1337)
        return aggregate_clusters(graph)
