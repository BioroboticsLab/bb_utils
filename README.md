# bb_utils

[![Documentation Status](https://readthedocs.org/projects/bb-utils/badge/?version=latest)](http://bb-utils.readthedocs.io/en/latest/?badge=latest)

This repository holds beesbook utilities.

Fiducial markers
================
The submodule `bb_utils.fiducial` holds helper functions to generate simple fiducial markers carrying one bit of information
as well as functions to find such markers in images.

Usage example:
```
# Load an arbitrary image that contains four fiducial markers in the corners.
# The approximate marker size must be known.
image = "/mnt/storage/david/data/beesbook/tagdetect/2018ld01.jpg"
markersize = 22
image = scipy.ndimage.imread(image)
# Find four markers in the image's corners (20% of the image's width/height).
# We are using the default markers here.
markers = bb_utils.fiducial.locate_markers_in_corners(image, markersize, corner_ratio=0.20)
# Match the recognized points against the homography of 2018.
H = bb_utils.fiducial.match_homography_points(markers)
# Display everything.
bb_utils.fiducial.display_markers(image, markers, H, dsize=(410, 300))
```

Output:
![image](https://user-images.githubusercontent.com/6689731/42174824-cb92710e-7e23-11e8-845d-0f9ab1bcbc1d.png)
