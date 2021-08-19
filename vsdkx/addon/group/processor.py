import numpy as np
from sklearn.cluster import AgglomerativeClustering


class GroupProcessor:
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
    """

    def __init__(self, distance_threshold, min_group_size):
        """
        Args:
            distance_threshold (int | float): Distance threshold required by
            the clustering algorithm
            min_group_size (int): Minimum amount of detected people to be
            considered a group
        """
        self.cluster = AgglomerativeClustering(
            affinity='euclidean',
            linkage='complete',
            compute_distances=False,
            distance_threshold=distance_threshold,
            n_clusters=None)
        self.min_group_size = min_group_size

    def get_centroids(self, boxes):
        """
        Calculates each boxes' centroid
        Args:
            boxes (np.array): Array with detected bounding boxes

        Returns:
            (list): List with centroids
            (np.array): Array with filtered boxes
        """

        centroids = []
        filtered_boxes = []

        for box in boxes:
            c = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
            centroids.append(c)
            filtered_boxes.append(box)

        return centroids, np.array(filtered_boxes)

    def get_groups(self, boxes, clusters, trackable_objects, centroids):
        """
        Creates groups out of the bounding boxes and their assigned cluster ID

        Args:
            boxes (np.array): Array with bounding boxes
            clusters (list): List with cluster IDs
            trackable_objects (dict): Dictionary of trackable objects
            centroids (tuple): Tuple with centroids

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
                # Group the boxes based on the walking direction of the object
                # This can result into three different groups
                # 1. People that walk 'upwards'
                # 2. People that walk 'downwards'
                # 3. People with an 'undefined' walking direction
                clustered_groups = \
                    self.group_by_direction(cluster_centroids,
                                            cluster_boxes,
                                            trackable_objects)
                # Iterate over the filtered cluster groups
                # and generate the group bounding box
                for cluster_group in clustered_groups:
                    group_box = np.array([min(cluster_group[:, 0]),
                                          min(cluster_group[:, 1]),
                                          max(cluster_group[:, 2]),
                                          max(cluster_group[:, 3])])
                    group_boxes.append(group_box)
                    people_count.append(len(cluster_group))

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
            box = boxes[idx[0]]
            centroid = centroids[idx[0]]
            cluster_boxes.append(box)
            centroids_list.append(centroid)
        return np.array(cluster_boxes), np.array(centroids_list)

    def post_process(self, boxes, trackable_objects):
        """
        Clusters the given bounding boxes to small clusters by their distance

        Args:
            boxes (np.array): Array with detected bounding boxes
            trackable_objects (dict): Dictionary with trackable objects
        Returns:
            (np.array): Array with cluster IDs
            (list): List with people counts per group
        """
        groups = []
        boxes = np.array(boxes)
        people_count = 0

        if len(boxes) > 1:
            # Get the bounding boxes centroids
            centroids, updated_boxes = self.get_centroids(boxes)
            print(f'Length of detected boxes {len(centroids)}'
                  f' length of trackable objects {len(trackable_objects)}')
            # Cluster the centroids
            y = self.cluster.fit(centroids)
            # Separate them into groups > self.min_group_size
            groups, people_count = self.get_groups(updated_boxes,
                                                   y.labels_,
                                                   trackable_objects,
                                                   centroids)

        return groups, people_count
