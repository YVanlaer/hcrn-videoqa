import h5py
import numpy as np

def extract_clips_with_consecutive_frames(h5data, num_clips, num_frames_per_clip):
    """
    Args:
        h5data: resnet feature of clip (num_frames) (dimension)
        num_clips: expected numbers of splitted clips
        num_frames_per_clip: number of frames in a single clip, pretrained model only supports 16 frames
    Returns:
        clips (num_clips) (num_frames_per_clip) (dimension)
    """
    clips = list()
    total_frames = h5data.shape[0]
    for i in np.linspace(0, total_frames, num_clips + 2, dtype=np.int32)[1:num_clips + 1]:
        clip_start = int(i) - int(num_frames_per_clip / 2)
        clip_end = int(i) + int(num_frames_per_clip / 2)
        if clip_start < 0:
            clip_start = 0
        if clip_end > total_frames:
            clip_end = total_frames - 1
        #if clip_end - clip_start != num_frames_per_clip:
        #    clip_end = clip_start - num_frames_per_clip
        clip = h5data[clip_start:clip_end]
        if clip_start == 0:
            shortage = num_frames_per_clip - (clip_end - clip_start)
            added_frames = []
            for _ in range(shortage):
                added_frames.append(np.expand_dims(h5data[clip_start], axis=0))
            if len(added_frames) > 0:
                added_frames = np.concatenate(added_frames, axis=0)
                clip = np.concatenate((added_frames, clip), axis=0)
        elif clip_end == (total_frames - 1):
            shortage = num_frames_per_clip - (clip_end - clip_start)
            added_frames = []
            for _ in range(shortage):
                added_frames.append(np.expand_dims(h5data[clip_end], axis=0))
            if len(added_frames) > 0:
                added_frames = np.concatenate(added_frames, axis=0)
                clip = np.concatenate((clip, added_frames), axis=0)
        clips.append(clip)
    return np.asarray(clips)

if __name__ == '__main__':
    with h5py.File('data/tv-qa/tv-qa_appearance_feat-clips.h5', 'w') as fd:
        with h5py.File('data/tv-qa/tv-qa_appearance_feat.h5', 'r') as app_features_file:
            #app_feat_id_to_index = {str(id): i for i, id in enumerate(app_features_file.file.keys())}
            for i,key in enumerate(app_features_file.file.keys()):
                clip = app_features_file[key]
                new_clips = extract_clips_with_consecutive_frames(clip, 6, 8)
                fd.create_dataset(key, data=new_clips)
