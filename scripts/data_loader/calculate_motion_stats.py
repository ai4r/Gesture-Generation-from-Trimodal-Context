import os
import lmdb
import numpy as np
import pyarrow

import utils.train_utils
import utils.data_utils


def calculate_data_mean(base_path):
    lmdb_path = os.path.join(base_path, 'lmdb_train')
    lmdb_env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with lmdb_env.begin() as txn:
        n_videos = txn.stat()['entries']
    src_txn = lmdb_env.begin(write=False)
    cursor = src_txn.cursor()

    pose_seq_list = []
    total_duration = 0

    for key, value in cursor:
        video = pyarrow.deserialize(value)
        vid = video['vid']
        clips = video['clips']
        for clip_idx, clip in enumerate(clips):
            poses = clip['skeletons_3d']
            pose_seq_list.append(poses)
            total_duration += (clip['end_time'] - clip['start_time'])

    # close db
    lmdb_env.close()

    all_poses = np.vstack(pose_seq_list)
    mean_pose = np.mean(all_poses, axis=0)

    # mean dir vec
    dir_vec = utils.data_utils.convert_pose_seq_to_dir_vec(all_poses)
    mean_dir_vec = np.mean(dir_vec, axis=0)

    # mean bone length
    bone_lengths = []
    for i, pair in enumerate(utils.data_utils.dir_vec_pairs):
        vec = all_poses[:, pair[1]] - all_poses[:, pair[0]]
        bone_lengths.append(np.mean(np.linalg.norm(vec, axis=1)))

    print('mean pose', repr(mean_pose.flatten()))
    print('mean directional vector', repr(mean_dir_vec.flatten()))
    print('mean bone lengths', repr(bone_lengths))
    print('total duration of the valid clips: {:.1f} h'.format(total_duration/3600))


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')
    np.set_printoptions(precision=7, suppress=True)

    lmdb_base_path = '../../data/ted_dataset'
    calculate_data_mean(lmdb_base_path)
