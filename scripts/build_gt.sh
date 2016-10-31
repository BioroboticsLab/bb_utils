#! /usr/bin/env bash

set -euo pipefail

VIDEO_FILE="/mnt/storage/beesbook/truth/video_files.txt"
IMAGE_FILE="/mnt/storage/beesbook/truth/image_directories.txt"

print_help () {
    echo "$0 DIRECORY"
    echo "creates a train/test split of the groud truth data."
    echo "Output is saved to DIRECORY"
}
if [ -z ${1+x} ]; then
    echo "No output directory is given."
    echo ""
    print_help
    exit 1
fi

if [ "$1" == "--help" ]; then
    print_help
    exit 0
fi

DATA_DIR="$1"

mkdir -p $DATA_DIR

echo_and_run() {
    echo "$@"
    $@
}

CMD="bb_gt_to_hdf5 --videos $VIDEO_FILE --images $IMAGE_FILE"

GT_14_0="/mnt/storage/beesbook/truth/20140805_Truth/repo_truth/2014/08/05/15/00/Cam_2_2014-08-05T15:17:00.100000Z--2014-08-05T15:17:59.400000Z.bbb"
GT_14_1="/mnt/storage/beesbook/truth/20140805_Truth/repo_truth/2014/08/05/15/00/Cam_0_2014-08-05T15:17:00.100000Z--2014-08-05T15:17:59.400000Z.bbb"

GT_15_0="/mnt/storage/beesbook/truth/20150922_Truth/repo_truth/2015/09/22/11/20/Cam_0_2015-09-22T11:36:34.006945Z--2015-09-22T11:42:15.024450Z.bbb"
GT_15_1="/mnt/storage/beesbook/truth/20150918_Truth/repo_truth/2015/09/18/09/20/Cam_0_2015-09-18T09:35:34.425149Z--2015-09-18T09:41:15.442654Z.bbb"
GT_15_2="/mnt/storage/beesbook/truth/20150918_Truth/repo_truth/2015/09/18/02/40/Cam_1_2015-09-18T02:51:38.554937Z--2015-09-18T02:57:19.572442Z.bbb"

GTs=($GT_14_0 $GT_14_1 $GT_15_0 $GT_15_1)

set +x
for gt in "${GTs[@]}"; do
    filename=${gt##*/}
    without_ext=${filename%.bbb}
    echo_and_run $CMD -g $gt  $DATA_DIR/$without_ext.hdf5
done

# train / test
echo_and_run $CMD -g ${GT_14_0} -g ${GT_15_0} -g ${GT_15_1} $DATA_DIR/gt_train.hdf5
echo_and_run $CMD -g ${GT_14_1} -g ${GT_15_2} $DATA_DIR/gt_test.hdf5

shuffle_hdf5 -o $DATA_DIR/gt_train_shuffled.hdf5 $DATA_DIR/gt_train.hdf5
shuffle_hdf5 -o $DATA_DIR/gt_test_shuffled.hdf5 $DATA_DIR/gt_test.hdf5
