import numpy as np
import unittest

from vsdkx.addon.group.Agglomerative import AgglomerativeGroupProcessor
from vsdkx.addon.group.DBSCAN import DBSCANGroupProcessor
from vsdkx.addon.group.ChineseWhispers import ChineseWhispersGroupProcessor

from vsdkx.addon.tracking.trackableobject import TrackableObject
from vsdkx.core.structs import AddonObject, Inference


class TestAddon(unittest.TestCase):
    addon_config = {
        "min_group_size": 1
    }

    def test_agglomerative(self):
        agglomerative = AgglomerativeGroupProcessor(self.addon_config, {}, {}, {})

        # first and last should be in same cluster
        boxes = np.array([[120, 150, 170, 200],
                          [100, 130, 150, 180],
                          [400, 300, 100, 100],
                          [122, 152, 170, 200]])

        inference = Inference()
        inference.boxes = boxes

        # test get_centroids
        centroids = agglomerative.get_centroids(boxes)

        self.assertEqual(centroids[0], (145, 175))
        self.assertEqual(centroids[1], (125, 155))
        self.assertEqual(centroids[2], (250, 200))
        self.assertEqual(centroids[3], (146, 176))

        # test bb_intersection_over_union
        iou_1 = agglomerative.bb_intersection_over_union(boxes[0], boxes[3])
        iou_2 = agglomerative.bb_intersection_over_union(boxes[0], boxes[0])
        iou_3 = agglomerative.bb_intersection_over_union(boxes[1], boxes[2])

        self.assertAlmostEqual(0.9, iou_1, places=1)
        self.assertEqual(1, iou_2)
        self.assertEqual(0, iou_3)

        # test _update_distance_threshold
        distance_threshold = agglomerative.distance_threshold
        agglomerative._update_distance_threshold(centroids)

        self.assertGreater(agglomerative.distance_threshold, distance_threshold * 0.95)
        agglomerative.distance_threshold = distance_threshold

        # test get_cluster_boxes
        indexes = [[0], [1]]
        cluster_boxes, cluster_centroids = agglomerative.get_cluster_boxes(boxes, indexes, centroids)

        self.assertEqual(2, len(cluster_boxes))
        self.assertTrue(self.check_array_contains(cluster_boxes, boxes[indexes[0]]))
        self.assertTrue(self.check_array_contains(cluster_boxes, boxes[indexes[1]]))

        trackable_object_0 = TrackableObject(0, centroids[0], boxes[0])
        trackable_object_1 = TrackableObject(1, centroids[1], boxes[1])
        trackable_object_2 = TrackableObject(2, centroids[2], boxes[2])
        trackable_object_3 = TrackableObject(3, centroids[3], boxes[3])

        shared = {
            "trackable_objects": {
                "0": trackable_object_0,
                "1": trackable_object_1,
                "2": trackable_object_2,
                "3": trackable_object_3
            }
        }

        # test post_process
        test_object = AddonObject(frame=[], inference=inference, shared=shared)
        result = agglomerative.post_process(test_object)

        groups = result.inference.extra['tracked_groups']
        objects = result.inference.extra['objects_in_groups']

        self.assertEqual(len(groups), len(objects))
        self.assertEqual(4, sum(objects))
        # should be min 2 elements in first cluster
        self.assertGreaterEqual(objects[0], 2)
        # should be max 3 clusters, min 1 cluster
        self.assertLessEqual(len(groups), 3)
        self.assertGreaterEqual(len(groups), 1)

    def test_dbscan(self):
        dbscan = DBSCANGroupProcessor(self.addon_config, {}, {}, {})

        boxes = np.array([[120, 150, 170, 200],
                          [100, 130, 150, 180],
                          [400, 300, 100, 100],
                          [122, 152, 170, 200]])
        centroids = dbscan.get_centroids(boxes)

        inference = Inference()
        inference.boxes = boxes

        trackable_object_0 = TrackableObject(0, centroids[0], boxes[0])
        trackable_object_0.direction = "up"
        trackable_object_1 = TrackableObject(1, centroids[1], boxes[1])
        trackable_object_1.direction = "undefined"
        trackable_object_2 = TrackableObject(2, centroids[2], boxes[2])
        trackable_object_2.direction = "undefined"
        trackable_object_3 = TrackableObject(3, centroids[3], boxes[3])
        trackable_object_3.direction = "down"

        shared = {
            "trackable_objects": {
                "0": trackable_object_0,
                "1": trackable_object_1,
                "2": trackable_object_2,
                "3": trackable_object_3
            }
        }

        # test group_by_direction
        cluster_groups = dbscan.group_by_direction(centroids, boxes, shared["trackable_objects"])

        self.assertEqual(len(cluster_groups[0]), 1)
        self.assertTrue(self.check_array_contains(cluster_groups[0], trackable_object_0.bounding_box))

        trackable_object_0.direction = "down"
        cluster_groups = dbscan.group_by_direction(centroids, boxes, shared["trackable_objects"])

        self.assertEqual(len(cluster_groups[0]), 2)
        self.assertTrue(self.check_array_contains(cluster_groups[0], trackable_object_0.bounding_box))
        self.assertTrue(self.check_array_contains(cluster_groups[0], trackable_object_3.bounding_box))

        trackable_object_0.direction = "undefined"
        trackable_object_3.direction = "undefined"
        cluster_groups = dbscan.group_by_direction(centroids, boxes, shared["trackable_objects"])

        self.assertEqual(len(cluster_groups[0]), 4)

        min_group_size = dbscan.min_group_size
        dbscan.min_group_size = 5
        cluster_groups = dbscan.group_by_direction(centroids, boxes, shared["trackable_objects"])
        dbscan.min_group_size = min_group_size

        self.assertEqual(len(cluster_groups), 0)

        # test get_groups
        n_clusters = 2
        clusters = [0, 0, 1, 1]
        group1_expected = [100, 130, 170, 200]
        group2_expected = [122, 152, 170, 200]
        groups, people_count = dbscan.get_groups(boxes, clusters, centroids)

        self.assertEqual(len(people_count), n_clusters)
        self.assertEqual(people_count[0], 2)
        self.assertEqual(people_count[1], 2)

        self.assertEqual(len(groups), 2)
        self.assertTrue(self.check_array_contains(groups, group1_expected))
        self.assertTrue(self.check_array_contains(groups, group2_expected))

        # test post_process
        test_object = AddonObject(frame=[], inference=inference, shared=shared)
        result = dbscan.post_process(test_object)

        groups = result.inference.extra['tracked_groups']
        objects = result.inference.extra['objects_in_groups']

        self.assertEqual(len(groups), len(objects))
        self.assertEqual(4, sum(objects))
        # should be min 2 elements in first cluster
        self.assertGreaterEqual(objects[0], 2)
        # should be max 3 clusters, min 1 cluster
        self.assertLessEqual(len(groups), 3)
        self.assertGreaterEqual(len(groups), 1)

    # This method checks if 2D array contains 1D array
    def check_array_contains(self, array2d, array):
        for ar in array2d:
            if np.all(ar == array):
                return True

        return False


if __name__ == '__main__':
    unittest.main()
