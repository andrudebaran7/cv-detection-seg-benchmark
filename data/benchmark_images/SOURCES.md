# Benchmark image set

Eight images from the COCO val2017 split, fetched by `benchmark/fetch_images.py`
from the official URL pattern `http://images.cocodataset.org/val2017/<id>.jpg`.
COCO images are licensed per the COCO terms (annotations CC BY 4.0; images per their
Flickr sources). Used here only for measuring inference latency/memory — no labels are
used and no accuracy is computed.

IDs: 139, 285, 632, 724, 776, 785, 802, 872
