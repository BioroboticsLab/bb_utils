import numpy as np
import matplotlib.pyplot as plt

def plot_marker(ax, inverse=False):
    """Plots a marker to an existing pyplot axis.

        Arguments:
            ax: pyplot.Axis object
            inverse: Whether the color of the marker will be inverted.
    """
    import matplotlib.patches

    k, w = "k", "w"
    if not inverse:
        k, w = w, k
    # Draws one marker.
    def rect(s, color):
        s /= 2.0
        ax.add_artist(matplotlib.patches.Rectangle([-s, -s], s*2, s*2, fc=color))
    r = 1.0
    rect(r * 1.1, color=w)
    rect(r, color=k)
    rect(r * 0.75, color=w)
    rect(r * 0.25, color=k)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axis("off")

def plot_sheet(w=4, h=4, figsize=(32, 32)):
    """Plots several markers into one pyplot figure that can be printed.

        Arguments:
            w, h: number of markers that the sheet will contain in its width and height.
            figsize: size of figure in inches; passed to plt.subplots
    """
    import itertools
    fig, axes = plt.subplots(w, h, figsize=figsize)
    axes = itertools.chain(*axes)

    for i, ax in enumerate(axes):
        plot_marker(ax, inverse=(i % 2 == 0))
    plt.show()

def generate_marker(inverse=False, save_to=None, resize=None):
    """Draws a marker using pyplot and then returns it as a numpy array.
        This involves storing the pyplot figure into a temporary file.

        Arguments:
            inverse: Whether black and white in the marker will be swapped.
            save_to: path or file-like object that the numpy array will be saved to in npz format.
            resize: integer. width of the rectangular marker (e.g. resize=512).
    """
    import tempfile
    import scipy.ndimage

    with tempfile.NamedTemporaryFile() as f:
        # Generate the marker using pyplot.
        # The marker will be saved to a temporary file and then loaded again.
        fig, ax = plt.subplots(1, 1, figsize=(16,16))
        plot_marker(ax, inverse=inverse)
        plt.savefig(f)
        plt.close()

        # Load the marker to have it as a numpy array.
        # The empty margin around the marker will be cropped.
        marker = scipy.ndimage.imread(f.name)[:,:,0]
        marker = (marker / 255.0).astype(np.float32)
        # Crop the empty borders.
        x, x2 = np.where(np.any(marker < 1.0, axis=0))[0][[0, -1]]
        y, y2 = np.where(np.any(marker < 1.0, axis=1))[0][[0, -1]]
        marker = marker[y:y2, x:x2]

        if resize is not None:
            import skimage.transform
            marker = skimage.transform.resize(marker, (resize, resize))
        if save_to is not None:
            np.savez_compressed(save_to, marker=marker)
        return marker

def get_marker(load=True, **kwargs):
    """Returns a default marker as a numpy array.
        The marker will be loaded from the python package by default.
        Keyword arguments will be passed to generate_marker.
    """
    if not load:
        return generate_marker(**kwargs)
    
    import pkg_resources
    marker_path = pkg_resources.resource_filename('bb_utils', 'data/fiducial_marker.npz')
    marker_file = np.load(marker_path)
    marker = marker_file["marker"]
    marker_file.close()

    return marker

def locate_markers(image, markersize, n_markers, marker=None, rescale="auto"):
    """Attempts to locate markers in an image. The markersize in pixels must be approximately known.
        Returns the first n_markers with the highest score.
        The image can be rescaled automatically to make the convolution faster.

        Arguments:
            image: Image to search (e.g. as returned by scipy.ndimage.imread).
            markersize: Approximate size of the markers in pixels.
            n_markers: Amount of markers to return.
            marker: (optional) numpy array; marker template. Will be loaded with get_marker() if not given.
            rescale: scaling factor for the image. "auto" means that the image will be scaled so that the markers are still sufficiently larger.

        Returns:
            list of (x, y, marker_type, score): x, y are pixel coordinates in the original image.
                                                marker_type (boolean) True if the marker's center is white.
                                                score: (float) arbitrary score of the marker's quality.
    """
    import skimage.feature
    import skimage.morphology

    import skimage.exposure
    import skimage.filters
    import scipy.signal
    
    if marker is None:
        marker = get_marker()

    if rescale == "auto":
        rescale = 25 / markersize
    
    # Rescale marker.
    marker = marker.astype(np.float32)
    marker = marker - marker.min()
    marker /= marker.max()
    marker = (marker - 0.5) * 2.0
    original_marker = marker

    # Rescale image.
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0
        
    if rescale != 1.0:
        image = skimage.transform.rescale(image, rescale)
        markersize = int(markersize * rescale)
    
    image = skimage.exposure.equalize_adapthist(image, kernel_size=markersize)
    image = image - image.min()
    image /= image.max()
    image = (image - 0.5) * 2.0    
    
    steps = 3
    rotation_steps = 3
    convs = None
    for step in range(steps):
        s = (step - steps // 2) * 2
        marker = skimage.transform.resize(original_marker, (markersize + s, markersize + s))
        
        for r in np.linspace(-5.0, +5.0, num=rotation_steps):
            if r == 0.0:
                rotated_marker = marker
            else:
                rotated_marker = skimage.transform.rotate(image = marker, angle=r, resize=True)
            conv = scipy.signal.convolve2d(image, rotated_marker, mode="same")
            conv2 = scipy.signal.convolve2d(image, -1.0 * rotated_marker, mode="same")

            conv = np.abs(conv) * np.abs(conv2)
            if convs is None:
                convs = conv
            else:
                idx = conv > convs
                convs[idx] = conv[idx]
            
    extrema = skimage.feature.peak_local_max(convs)
    extrema = [(convs[y, x] / (rotation_steps * steps), x, y) for (y, x) in extrema]
    extrema = sorted(extrema, reverse=True)
    
    results = []
    for (score, x, y) in extrema[:n_markers]:
        sub_image = image[(y-2):(y+2), (x-2):(x+2)]
        marker_type = sub_image.sum() > 0.0
        results.append((x / rescale, y / rescale, marker_type, score))
    return results

def locate_markers_in_corners(image, markersize, marker=None, corner_ratio=0.25, **kwargs):
    """Searches the four corners of an image for markers and returns four markers.
        
    Arguments:
        image: The image to pass to locate_markers.
        markersize: Approximate size of markers in the image in pixels.
        marker: (optional) Marker template.
        corner_ratio: fraction of image width and height to search for a marker in each corner.

    Returns:
        list of (x, y, marker_type, score)x4: x, y are pixel coordinates in the original image.
                                            marker_type (boolean) True if the marker's center is white.
                                            score: (float) arbitrary score of the marker's quality.
    """
    if len(image.shape) > 2:
        import skimage.color
        image = skimage.color.rgb2hsv(image)[:,:,2]

    h, w = image.shape
    corner_w = int(image.shape[1] * corner_ratio)
    corner_h = int(image.shape[0] * corner_ratio)
    
    results = []
    for (x, y) in ((0, 0), (w - corner_w, 0), (0, h - corner_h), (w - corner_w, h - corner_h)):
        sub_image = image[y:(y + corner_h), x:(x + corner_w)]
        r = locate_markers(sub_image, markersize=markersize, marker=marker, n_markers=1, **kwargs)
        x_, y_, t, s = r[0]
        x_ += x
        y_ += y
        results.append((x_, y_, t, s))
    return results

def arg_clockwise_order(pts):
    """Returns the indices of the four points in pts in clockwise order starting from top-left.
    """
    s = pts.sum(axis = 1)
    diff = np.diff(pts, axis = 1)
    return [np.argmin(s),
            np.argmin(diff),
            np.argmax(s),
            np.argmax(diff)]

def match_homography_points(points, scale=10.0, year=2018):
    """Takes recognized markers, treats them as corner points of a homography and returns a matching homography matrix.
        The four points must contain either exactly three points with type=False or type=True. This is necessary for sorting them correctly.

        Arguments:
            points: list of (x, y, type, score)x4.
            scale: Arbitrary scale that is multiplied to the homography (1 = cm, 10 = mm).
            year: The year of the homography.
    """
    if len(points) != 4:
        raise ValueError("List must be of length 4.")

    import cv2

    if year == 2018:
        # The points were measured manually with love.
        # They are in centimeters and start from the top-left corner in clockwise direction.
        # The first side had three white markers and one black marker (vice-verse on the other side).
        # The third point was of the inverse marker type compared to the other three.
        homography_points = [((0, 0), (40.4, -0.1), (40.6, 27.9), (-0.1, 27.9)),
                        ((0, 0),  (40.2, 0), (40.6, 28.1), (0.4, 28.1))] 
    else:
        raise ValueError("Homography data only available for 2018.")
    # Extract XY coordinates from the points (image pixel coordinates).
    xy = np.array([(p[0], p[1]) for p in points])
    # Extract the marker types from the points (either three True and one False or vice-versa).
    types = np.array([p[2] for p in points])
    # The amount of white markers detemines the side (either 1 or 3).
    image_side = 0 if np.sum(types) > 2 else 1
    high_id = [False, True][image_side] # "Special" marker
    # Order the points in clockwise order.
    order = arg_clockwise_order(xy)
    resorted_types = types[order]
    high_id_idx = np.argwhere(resorted_types == high_id)[0][0]
    # Shift the atypical marker to the third position of the array.
    shift = 4 + high_id_idx - 2
    xy = xy[order, :] # Clockwise.
    xy = np.roll(xy, shift=shift, axis=0) # And shifted.
    # The points the homography will map the markers to. Scaled by an arbitrary factor (e.g. for debugging).
    target_points = np.array(homography_points[image_side]) * scale
    
    H, _ = cv2.findHomography(xy, target_points)
    return H

def display_markers(image, markers, homography=None, figsize=(20, 8), dsize=None):
    """Helper function to display the recognized markers in an image.
        Can optionally take and apply a homography matrix H.

        Arguments:
            image: Image to show.
            markers: Recognized markers as returned by locate_markers.
            homography: (optional) H matrix of a homography.
            figsize: Size of the figure in inches; passed to pyplot.
            dsize: Size of the image to draw the transformed image on (if homography is given).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.imshow(image, cmap="gray")
    for (x, y, t, score) in markers:
        ax1.scatter(x, y, c=["b", "r"][int(t)], marker="s")
    if homography is not None:
        import cv2
        if dsize is None:
            dsize = image.shape[:2][::-1]
        image2 = cv2.warpPerspective(image, homography, dsize=dsize)
        ax2.imshow(image2, cmap="gray")
    plt.show()