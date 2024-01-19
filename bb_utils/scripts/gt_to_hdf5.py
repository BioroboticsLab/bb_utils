from __future__ import print_function

import numpy as np
from bb_binary import int_id_to_binary, parse_video_fname, \
    convert_frame_to_numpy, parse_image_fname, load_frame_container
from pipeline.io import raw_frames_generator
from pipeline.stages.visualization import ResultCrownVisualizer
from pipeline.stages.processing import Localizer
from diktya.numpy import tile, image_save

from diktya.distributions import DistributionCollection, Bernoulli
from deepdecoder.data import DistributionHDF5Dataset
import click
import os
from scipy.misc import imread
from datetime import timedelta


class GTPeriod:
    def __init__(self, camIdx, start, end, filename, frames):
        self.camIdx = camIdx
        self.start = start
        self.end = end
        self.filename = filename
        self.frames = frames

    def pickle_filename(self):
        basename, ext = os.path.splitext(os.path.basename(self.filename))
        return basename + ".pickle"


def get_subdirs(dir):
    return listdir(dir, os.path.isdir)


def get_files(dir):
    return listdir(dir, os.path.isfile)


def listdir(dir, select_fn):
    return [os.path.join(dir, d) for d in os.listdir(dir)
            if select_fn(os.path.join(dir, d))]


def extract_gt_rois(data, image, date, roi_size=128):
    if date.year == 2014:
        pos = np.stack([data['ypos'], data['xpos']], axis=1)
    else:
        pos = np.stack([data['xpos'], data['ypos']], axis=1)

    rois, mask = Localizer.extract_rois(pos, image, roi_size)
    return rois, mask, pos


def drop_microseconds(ts):
    return ts.replace(microsecond = 0)


class FrameGeneratorFactory():
    def __init__(self, video_files, image_dirs):
        self.videos = video_files
        self.image_dirs = image_dirs
        self.sources = {parse_video_fname(video)[:2]: ('video', video) for video in self.videos}
        for dir in self.image_dirs:
            first_frame = sorted(get_files(dir))[0]
            parsed = parse_image_fname(first_frame)
            self.sources[parsed] = ('image', dir)

        self.sources = {(cam, drop_microseconds(ts)): val
                        for (cam, ts), val in self.sources.items()}

    def get_generator(self, camIdx, startts):
        source_type, source_fname = self.sources[(camIdx, drop_microseconds(startts))]
        if source_type == 'video':
            for x in raw_frames_generator(source_fname):
                yield x, source_fname
        elif source_type == 'image':
            for img_fname in sorted(get_files(source_fname)):
                yield imread(img_fname), img_fname
        else:
            raise Exception("Unknown source: {}".format(source_type))


def append_gt_to_hdf5(gt_period, dset):
    for gt_frame in gt_period.frames:
        h5_frame = {k: v for k, v in gt_frame.items() if type(v) == np.ndarray}
        dist = dset.get_tag_distribution()
        labels = np.zeros((len(h5_frame['bits']),), dtype=dist.norm_dtype)
        labels['bits'] = h5_frame.pop('bits')
        dset.append(labels=labels, **h5_frame)


@click.command("bb_gt_to_hdf5")
@click.option('--gt-file', '-g', help='file with bb binary *.capnp filenames',
              type=click.Path(), required=True, multiple=True)
@click.option('--videos', '-v', help='video filename', type=click.File(), required=False)
@click.option('--images', '-i', help='file with image directories', type=click.File(), required=False)
@click.option('--visualize-debug', is_flag=True)
@click.option('--fix-utc-2014', type=bool, default=True)
@click.argument('output')
def run(gt_file, videos, images, visualize_debug, output, fix_utc_2014, nb_bits=12):
    """
    Converts bb_binary ground truth Cap'n Proto files to hdf5 files and
    extracts the corresponding rois from videos or images.
    """
    def get_filenames(f):
        if f is None:
            return []
        else:
            return [line.rstrip('\n') for line in f.readlines()]

    gen_factory = FrameGeneratorFactory(get_filenames(videos),
                                        get_filenames(images))
    if os.path.exists(output):
        os.remove(output)

    distribution = DistributionCollection([('bits', Bernoulli(), nb_bits)])
    dset = DistributionHDF5Dataset(output, distribution)
    camIdxs = []
    periods = []
    for fname in gt_file:
        fc = load_frame_container(fname)
        camIdx, start_dt, end_dt = parse_video_fname(fname)
        if fix_utc_2014 and start_dt.year == 2014:
            start_dt -= timedelta(hours=2)
        gt_frames = []
        gen = gen_factory.get_generator(camIdx, start_dt)
        for frame, (video_frame, video_filename) in zip(fc.frames, gen):
            gt = {}
            np_frame = convert_frame_to_numpy(frame)
            rois, mask, positions = extract_gt_rois(np_frame, video_frame, start_dt)
            for name in np_frame.dtype.names:
                gt[name] = np_frame[name][mask]
            bits = [int_id_to_binary(id)[::-1] for id in gt["decodedId"]]
            gt["bits"] = 2*np.array(bits, dtype=float) - 1
            gt["tags"] = 2 * (rois / 255.).astype(np.float16) - 1
            gt['filename'] = os.path.basename(video_filename)
            gt['camIdx'] = camIdx
            gt_frames.append(gt)
            print('.', end='', flush=True)
        print()
        gt_period = GTPeriod(camIdx, start_dt, end_dt, fname, gt_frames)

        periods.append([int(gt_period.start.timestamp()), int(gt_period.end.timestamp())])
        camIdxs.append(gt_period.camIdx)
        append_gt_to_hdf5(gt_period, dset)

    dset.attrs['periods'] = np.array(periods)
    dset.attrs['camIdxs'] = np.array(camIdxs)
    visualize_detection_tiles(dset, os.path.splitext(output)[0])
    dset.close()


def visualize_detection_tiles(dset, name, n=20**2):
    crown_vis = ResultCrownVisualizer()
    imgs_raw = []
    imgs_overlayed = []
    nb_tags = len(dset['tags'])
    n = min(nb_tags, n)
    indicies = np.arange(nb_tags)
    np.random.shuffle(indicies)
    for i in range(n):
        p = indicies[i]
        position = np.array([dset['tags'][0, 0].shape]) / 2
        tag = dset['tags'][p, 0]
        bits = (dset['bits'][p:p+1] + 1) / 2.
        overlay = crown_vis(tag, position, np.zeros((1, 3)), bits)[0]
        overlayed = crown_vis.add_overlay((tag+1)/2, overlay)
        imgs_raw.append(tag)
        imgs_overlayed.append(2*overlayed - 1)

    tiled_raw = tile([img.swapaxes(0, -1) for img in imgs_raw])
    tiled_overlayed = tile([img.swapaxes(0, -1) for img in imgs_overlayed])
    image_save(name + '_raw.png', tiled_raw)
    image_save(name + '_overlayed.png', tiled_overlayed)
