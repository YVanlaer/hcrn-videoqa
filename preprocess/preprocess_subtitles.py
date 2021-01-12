import argparse
import numpy as np
import os
import re
import json
import nltk
from datautils import utils
import pandas as pd
import pickle

def subtitle_encoding_data(args, vocab, video_names, subtitles, mode='train'):
    # Encode all subtitles
    print('Encoding data')
    subtitles_encoded = []
    subtitles_len = []
    video_names_tbw = []
    for idx, subtitle in enumerate(subtitles):
        subtitle = subtitle.lower()[:-1]
        subtitle_tokens = nltk.word_tokenize(subtitle)
        # truncate at 256 tokens
        subtitle_tokens = subtitle_tokens[:min(len(subtitle_tokens),256)]
        subtitle_encoded = utils.encode(subtitle_tokens, vocab['subtitle_token_to_idx'], allow_unk=True)
        subtitles_encoded.append(subtitle_encoded)
        subtitles_len.append(len(subtitle_encoded))
        video_names_tbw.append(video_names[idx])

    # Pad encoded subtitles
    max_subtitle_length = max(len(x) for x in subtitles_encoded)
    for se in subtitles_encoded:
        while len(se) < max_subtitle_length:
            se.append(vocab['subtitle_token_to_idx']['<NULL>'])

    subtitles_encoded = np.asarray(subtitles_encoded, dtype=np.int32)
    subtitles_len = np.asarray(subtitles_len, dtype=np.int32)
    print(subtitles_encoded.shape)

    glove_matrix = None
    if mode in ['train']:
        token_itow = {i: w for w, i in vocab['subtitle_token_to_idx'].items()}
        print("Load glove from %s" % args.glove_pt)
        #glove = pickle.load(open(args.glove_pt, 'rb'))
        df = pd.read_csv(args.glove_pt, sep =" ", quoting=3, header=None, index_col=0)
        print("glove file loaded!")
        glove = {key: val.values for key, val in df.T.items()}
        dim_word = glove['the'].shape[0]
        glove_matrix = []
        for i in range(len(token_itow)):
            vector = glove.get(token_itow[i], np.zeros((dim_word,)))
            glove_matrix.append(vector)
        glove_matrix = np.asarray(glove_matrix, dtype=np.float32)
        print(glove_matrix.shape)

    print('Writing ', args.output_pt.format(args.dataset, args.dataset, mode))
    obj = {
        'subtitles': subtitles_encoded,
        'subtitles_len': subtitles_len,
        'video_names': np.array(video_names_tbw),
        'glove': glove_matrix,
    }
    with open(args.output_pt.format(args.dataset, args.dataset, mode), 'wb') as f:
        pickle.dump(obj, f)

def process_subtitles(args):
    print('Loading data')
    with open(args.annotation_file, 'r') as dataset_file:
        video_names = []
        for line in dataset_file:
            instance = json.loads(line)
            video_names += [instance['vid_name']]

    # Load vocab from disk (question must have been preprocessed before)
    print('Loading vocab')
    with open(args.vocab_json.format(args.dataset, args.dataset), 'r') as f:
        vocab = json.load(f)
    if args.mode in ['train']:
        print('Building more vocab')
        subtitle_token_to_idx = {'<NULL>': 0, '<UNK>': 1}

    subtitles = []
    for video_name in video_names:
        print(video_name)
        with open(args.subtitles_folder.format(video_name) + '.srt', encoding="utf8") as f:
            content = f.read()
            capture = re.findall(r"\d+\n[0-9:,]+ --> [0-9:,]+\n((?:.+\n)+)\n", content)
            subtitle = " ".join([c.replace('\n', ' ') for c in capture])
        subtitles += [subtitle]

        if args.mode in ['train']:
            subtitle = subtitle.lower()
            tokens = nltk.word_tokenize(subtitle)
            for token in tokens[:min(len(tokens),256)]:
                if token not in subtitle_token_to_idx:
                    subtitle_token_to_idx[token] = len(subtitle_token_to_idx)

    if args.mode in ['train']:
        print('Get subtitle_token_to_idx')
        print(len(subtitle_token_to_idx))

        vocab['subtitle_token_to_idx'] = subtitle_token_to_idx

        print('Write into %s' % args.vocab_json.format(args.dataset, args.dataset))
        with open(args.vocab_json.format(args.dataset, args.dataset), 'w') as f:
            json.dump(vocab, f, indent=4)
    subtitle_encoding_data(args, vocab, video_names, subtitles, mode=args.mode)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='tgif-qa', choices=['tgif-qa', 'msrvtt-qa', 'msvd-qa', 'tv-qa'], type=str)
    parser.add_argument('--answer_top', default=4000, type=int)
    parser.add_argument('--glove_pt',
                        help='glove pickle file, should be a map whose key are words and value are word vectors represented by numpy arrays. Only needed in train mode')
    parser.add_argument('--output_pt', type=str, default='../data/{}/{}_{}_subtitles.pt')
    parser.add_argument('--vocab_json', type=str, default='../data/{}/{}_vocab.json')
    parser.add_argument('--mode', choices=['train', 'val', 'test'])
    parser.add_argument('--seed', type=int, default=666)

    args = parser.parse_args()
    np.random.seed(args.seed)

    args.subtitles_folder = 'C:/Users/youval/Downloads/tvqa_subtitles/{}'
    args.annotation_file = 'C:/Users/youval/Downloads/tvqa_qa_release/tvqa_{}.jsonl'.format(args.mode)
    # check if data folder exists
    if not os.path.exists('../data/{}'.format(args.dataset)):
        os.makedirs('../data/{}'.format(args.dataset))
    process_subtitles(args)