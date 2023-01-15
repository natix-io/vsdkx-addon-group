# Group Detection

This add on utilizes a clustering algorithm to perform group detection as a `post_process` step. 

## Features and Clustering algorithms

To tackle group detection we are extracting the following features from the detected bounding boxes:

- Bounding box centroids
- Bounding box width and height
- Bounding box distance from the camera
- Identity Matrix with IOUs per bounding box
- `Arctan2` of bounding box centroids

We have tested the stated features on the following three algorithms:

- Agglomerative 
- DBSCAN
- ChineseWhispers

Each algorithm is finetuned to run on different hyperparameters. By default we use DBSCAN as it has shown the best results, with ChineseWhispers achieving the poorest results both in speed and cluster performance.

## Addon Config

```yaml
group:
    class: vsdkx.addon.group.DBSCAN.DBSCANGroupProcessor
    min_group_size: 3
```

## Debug 

Example of object initialization and `post_process` step:

```python
from vsdkx.addon.group import DBSCANGroupProcessor
from vsdkx.core.interfaces import Addon, AddonObject

add_on_config = {
  'min_group_size': 3, 
   'class': 'vsdkx.addon.group.DBSCAN.DBSCANGroupProcessor'
   }
    
model_config = {
    'classes_len': 1, 
    'filter_class_ids': [0], 
    'input_shape': [640, 640], 
    'model_path': 'vsdkx/weights/ppl_detection_retrain_training_2.pt'
    }
    
model_settings = {
    'conf_thresh': 0.5, 
    'device': 'cpu', 
    'iou_thresh': 0.4
    }  
    
group_processor = DBSCANGroupProcessor(addon_on_config, model_settings, model_config)

addon_object = AddonObject(
  frame=np.array(RGB image),
  inference=Inference(
    boxes=[array([2007,  608, 3322, 2140]), array([ 348,  348, 2190, 2145])], 
    classes=[array([0], dtype=object), array([0], dtype=object)], 
    scores=[array([0.799637496471405], dtype=object), array([0.6711544394493103], dtype=object)], 
    extra={
      'tracked_objects': 0, 
      'zoning': {
        'zone_0': {
          'Person': [], 
          'Person_count': 0, 
          'objects_entered': {'Person': [], 'Person_count': 0}, 
          'objects_exited': {'Person': [], 'Person_count': 0}
          }, 
        'rest': {
          'Person': [],
          'Person_count': 0
        }
      }
     }
    ),
    shared={
      'trackable_objects': {}, 
      'trackable_objects_history': {
          0: {'object_id': 0}, 
          1: {'object_id': 1}
      }
    }
   )
   
addon_object = group_processor.post_process(addon_object)
```

The resulted `addon_object` after the `post_process` step is executed is:

```python

addon_object = AddonObject(
  frame=np.array(RGB image),
  inference=Inference(
    boxes=[array([2007,  608, 3322, 2140]), array([ 348,  348, 2190, 2145])], 
    classes=[array([0], dtype=object), array([0], dtype=object)], 
    scores=[array([0.799637496471405], dtype=object), array([0.6711544394493103], dtype=object)], 
    extra={
      'tracked_objects': 0, 
      'zoning': {
        'zone_0': {
          'Person': [], 
          'Person_count': 0, 
          'objects_entered': {'Person': [], 'Person_count': 0}, 
          'objects_exited': {'Person': [], 'Person_count': 0}
          }, 
        'rest': {
          'Person': [],
          'Person_count': 0
        },
        'tracked_groups': [], 
        'objects_in_groups': 0
      }
     }
    ),
    shared={
      'trackable_objects': {}, 
      'trackable_objects_history': {
          0: {'object_id': 0}, 
          1: {'object_id': 1}
      }
    }
   )
   
```

where 

```python
addon_object.inference.extra['tracked_groups'] = groups # bounding box coordinates of detected group
addon_object.inference.extra['objects_in_groups'] = people_count # people count per group
```
