import networkx as nx
import numpy as np
from chinese_whispers import chinese_whispers, aggregate_clusters
from scipy.spatial import distance as dist
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from vsdkx.core.interfaces import Addon
from vsdkx.core.structs import AddonObject


class GroupProcessor(Addon):
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
        metric (string): Metric name which is used for clustering functions
    """

    def __init__(self, addon_config: dict, model_settings: dict,
                 model_config: dict, drawing_config: dict):
        super().__init__(addon_config, model_settings, model_config,
                         drawing_config)

        self.clustering_algorithm = addon_config['algorithm']
        self.min_group_size = addon_config['min_group_size']
        self.temporal_len = 6
        self.feat_size = (self.temporal_len * 2) + 3
        self.metric = 'euclidean'

        if self.clustering_algorithm == 'dbscan':
            self.distance_threshold = 0.5

        elif self.clustering_algorithm == 'agglomerative':
            self.distance_threshold = 0.2

        elif self.clustering_algorithm == 'chinesewhispers':
            self.distance_threshold = 0.2

    def dbscan_clustering(self,
                          centroids=None,
                          distance_threshold_update=True):
        """
        Inits DBSCANClustering & Updates distance threshold

        Returns:
            cluster (DBSCANClustering): Clustering object
        """

        if distance_threshold_update:
            self.update_distance_threshold(centroids=centroids)

        cluster = DBSCAN(eps=self.distance_threshold,
                         min_samples=1,
                         metric=self.metric)

        return cluster

    def update_distance_threshold(self, centroids):
        """
        Updates distance threshold

        Args:
            centroids (np.array): Array with centroids
        """
        distances = np.unique(
            dist.cdist(centroids, centroids))

        distance_threshold = abs(distances.mean() - distances.std())

        self.distance_threshold = distance_threshold * 0.05 + \
                                  self.distance_threshold * 0.95

    @staticmethod
    def bb_intersection_over_union(box_a, box_b):
        """
        Finds the intersection over union between two bounding boxes
        Args:
            box_a (np.array): Array with bounding box coordinates
            box_b (np.array): Array with bounding box coordinates

        Returns:
            iou (float): Intersection over union score
        """
        # determine the (x, y)-coordinates of the intersection rectangle
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])
        # compute the area of intersection rectangle
        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
        box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = inter_area / float(box_a_area + box_b_area - inter_area)
        # return the intersection over union value

        return iou

    def get_features(self, boxes, trackable_objects):
        """
        Calculates each boxes' centroid
        Args:
            boxes (np.ndarray): List of bounding boxes
            trackable_objects (TrackableObjects): List with trackable objects

        Returns:
            (np.ndarray): List with centroids
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

        if self.clustering_algorithm == 'chinesewhispers':
            features = dist.cdist(features, features, metric=self.metric)

        features = MinMaxScaler().fit_transform(features)

        if self.clustering_algorithm == 'chinesewhispers':
            features[features > self.distance_threshold] = 0

        return features, centroids[:, 4:6], np.array(filtered_boxes)

    def get_groups(self, boxes, clusters, centroids):
        """
        Creates groups out of the bounding boxes and their assigned cluster ID

        Args:
            boxes (np.array): Array with bounding boxes
            clusters (np.array): List with cluster IDs
            centroids (np.array): Tuple with centroids

        Returns:
            (list): List of group bounding box
            (list): List with people counts per group
        """
        group_boxes = []
        people_count = []

        unique_clusters = np.unique(clusters)

        # Iterate over all unique cluster IDs and get all
        # IDX that belong to the same cluster
        for cluster_id in unique_clusters:
            idx = np.argwhere(cluster_id == clusters)

            # If the filtered IDXs have a length equal or bigger
            # than the min_group_size threshold, we treat this a new group
            if len(idx) >= self.min_group_size:
                # Get the bounding boxes that correspond to that group
                cluster_boxes, cluster_centroids = \
                    self.get_cluster_boxes(boxes, idx, centroids)

                if self.clustering_algorithm == 'chinesewhispers':
                    cluster_boxes = np.squeeze(cluster_boxes, axis=1)

                group_box = np.array([min(cluster_boxes[:, 0]),
                                      min(cluster_boxes[:, 1]),
                                      max(cluster_boxes[:, 2]),
                                      max(cluster_boxes[:, 3])])
                group_boxes.append(group_box)
                people_count.append(len(cluster_boxes))

        return group_boxes, people_count

    def group_by_direction(self, centroids, boxes, trackable_objects):
        """
        It groups the bounding boxes based on the walking direction of the
        object. This can result into the following three groups:
        1. People that walk 'upwards'
        2. People that walk 'downwards'
        3. People with an 'undefined' walking direction

        Args:
            centroids (tuple): Tuple with x,y values of a centroid
            boxes (np.array): Array with bounding boxes
            trackable_objects (dict): Dictionary with trackable objects

        Returns:
            (list): List of cluster groups
        """
        direction_up = []
        direction_down = []
        direction_undefined = []
        cluster_groups = []
        for box, centroid in zip(boxes, centroids):
            for to_idx in trackable_objects:
                to = trackable_objects[to_idx]
                to_centroid = to.centroids[-1]
                if np.array_equal(centroid, to_centroid):
                    object_direction = to.direction
                    if object_direction == 'up':
                        direction_up.append(box)
                    elif object_direction == 'down':
                        direction_down.append(box)
                    else:
                        direction_undefined.append(box)
                    break

        if len(direction_up) >= self.min_group_size:
            cluster_groups.append(np.array(direction_up))
        elif len(direction_down) >= self.min_group_size:
            cluster_groups.append(np.array(direction_down))
        elif len(direction_undefined) >= self.min_group_size:
            cluster_groups.append(np.array(direction_undefined))

        return cluster_groups

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
            if self.clustering_algorithm == 'chinesewhispers':
                box = boxes[idx]
                centroid = centroids[idx]
            else:
                box = boxes[idx[0]]
                centroid = centroids[idx[0]]
            cluster_boxes.append(box)
            centroids_list.append(centroid)

        return np.array(cluster_boxes), np.array(centroids_list)

    def agglomerative_clustering(self,
                                 centroids=None,
                                 distance_threshold_update=True):
        """
        Inits AgglomerativeClustering & Updates distance threshold

        Returns:
            cluster (AgglomerativeClustering): Clustering object
        """

        if distance_threshold_update:
            distances = np.unique(
                dist.cdist(centroids, centroids))

            distance_threshold = abs(distances.mean() - distances.std())

            self.distance_threshold = distance_threshold * 0.05 + \
                                      self.distance_threshold * 0.95

        cluster = AgglomerativeClustering(
            affinity=self.metric,
            linkage='ward',
            compute_distances=False,
            distance_threshold=self.distance_threshold,
            n_clusters=None)

        return cluster

    @staticmethod
    def get_centroids(boxes):
        """
        Calculates each boxes' centroid
        Args:
            boxes (np.array): Array with detected bounding boxes

        Returns:
            (list): List with centroids
        """

        centroids = []

        for box in boxes:
            c = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
            centroids.append(c)

        return centroids

    @staticmethod
    def extract_cluster_ids(prediction_dict: dict) -> np.ndarray:
        '''
        Extracts cluster ids from ChinesWhispers algorithm's prediction

        Args:
            prediction_dict: dict with predictions

        Returns:
            (np.ndarray): list with cluster ids
        '''
        cluster_ids = []
        for key, values in prediction_dict.items():
            for _ in range(len(values)):
                cluster_ids.append(key)

        return np.array(cluster_ids)

    def post_process(self, addon_object: AddonObject) -> AddonObject:
        """
        Clusters the given bounding boxes to small clusters by their distance

        Args:
            addon_object (AddonObject):
        Returns:
            (AddonObject): addon object has updated information for inference result
        """
        groups = []
        people_count = 0
        boxes = np.array(addon_object.inference.boxes)
        temporal_data = self.temporal_len * 2
        trackable_objects = addon_object.shared["trackable_objects"]

        if len(boxes) > 1:
            # Get the bounding boxes centroids
            features, centroids, boxes = self.get_features(boxes,
                                                           trackable_objects)

            if self.clustering_algorithm == 'dbscan':
                self.cluster = self.dbscan_clustering(
                    centroids=features[:, temporal_data - 2:temporal_data],
                    distance_threshold_update=True)
                # Cluster the centroids
                y = self.cluster.fit(features)
                # Separate them into groups > self.min_group_size
                groups, people_count = self.get_groups(boxes,
                                                       y.labels_,
                                                       centroids)

            elif self.clustering_algorithm == 'agglomerative':
                self.cluster = self.agglomerative_clustering(
                    centroids=features[:, temporal_data - 2:temporal_data],
                    distance_threshold_update=True)
                # Cluster the centroids
                y = self.cluster.fit(features)
                # Separate them into groups > self.min_group_size
                groups, people_count = self.get_groups(boxes,
                                                       y.labels_,
                                                       centroids)

            elif self.clustering_algorithm == 'chinesewhispers':
                self.update_distance_threshold(
                    centroids=features[:, temporal_data - 2:temporal_data])

                print(f'Length of detected boxes {len(centroids)}'
                      f' length of trackable objects {len(trackable_objects)}')
                # Cluster the centroids
                G = nx.from_numpy_matrix(features)
                chinese_whispers(G, seed=1337)
                y = aggregate_clusters(G)
                # Separate them into groups > self.min_group_size
                labels = self.extract_cluster_ids(y)
                groups, people_count = self.get_groups(boxes,
                                                       labels,
                                                       centroids)

            print(f'Length of detected boxes {len(centroids)}'
                  f' length of trackable objects {len(trackable_objects)}')
        addon_object.inference.extra['tracked_groups'] = groups
        addon_object.inference.extra['objects_in_groups'] = people_count

        return addon_object
