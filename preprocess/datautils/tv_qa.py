import os
import json
from datautils import utils
import nltk
import pandas as pd

import pickle
import numpy as np

def multichoice_encoding_data(args, vocab, questions, video_names, q_idx, answers, ans_candidates, mode='train'):
    # Encode all questions
    print('Encoding data')
    questions_encoded = []
    questions_len = []
    question_ids = []
    all_answer_cands_encoded = []
    all_answer_cands_len = []
    video_names_tbw = []
    correct_answers = []
    for idx, question in enumerate(questions):
        question = question.lower()[:-1]
        question_tokens = nltk.word_tokenize(question)
        question_encoded = utils.encode(question_tokens, vocab['question_answer_token_to_idx'], allow_unk=True)
        questions_encoded.append(question_encoded)
        questions_len.append(len(question_encoded))
        question_ids.append(q_idx[idx])
        video_names_tbw.append(video_names[idx])
        # grounthtruth
        answer = int(answers[idx])
        correct_answers.append(answer)
        # answer candidates
        candidates = ans_candidates[idx]
        candidates_encoded = []
        candidates_len = []
        for ans in candidates:
            ans = ans.lower()
            ans_tokens = nltk.word_tokenize(ans)
            cand_encoded = utils.encode(ans_tokens, vocab['question_answer_token_to_idx'], allow_unk=True)
            candidates_encoded.append(cand_encoded)
            candidates_len.append(len(cand_encoded))
        all_answer_cands_encoded.append(candidates_encoded)
        all_answer_cands_len.append(candidates_len)

    # Pad encoded questions
    max_question_length = max(len(x) for x in questions_encoded)
    for qe in questions_encoded:
        while len(qe) < max_question_length:
            qe.append(vocab['question_answer_token_to_idx']['<NULL>'])

    questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
    questions_len = np.asarray(questions_len, dtype=np.int32)
    print(questions_encoded.shape)

    # Pad encoded answer candidates
    max_answer_cand_length = max(max(len(x) for x in candidate) for candidate in all_answer_cands_encoded)
    for ans_cands in all_answer_cands_encoded:
        for ans in ans_cands:
            while len(ans) < max_answer_cand_length:
                ans.append(vocab['question_answer_token_to_idx']['<NULL>'])
    all_answer_cands_encoded = np.asarray(all_answer_cands_encoded, dtype=np.int32)
    all_answer_cands_len = np.asarray(all_answer_cands_len, dtype=np.int32)
    print(all_answer_cands_encoded.shape)

    glove_matrix = None
    if mode in ['train']:
        token_itow = {i: w for w, i in vocab['question_answer_token_to_idx'].items()}
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
        'questions': questions_encoded,
        'questions_len': questions_len,
        'question_id': question_ids,
        'video_names': np.array(video_names_tbw),
        'ans_candidates': all_answer_cands_encoded,
        'ans_candidates_len': all_answer_cands_len,
        'answers': correct_answers,
        'glove': glove_matrix,
    }
    if args.mode in ['test']:
        del obj['answers']
    with open(args.output_pt.format(args.dataset, args.dataset, mode), 'wb') as f:
        pickle.dump(obj, f)

def process_questions_mulchoices(args):
    print('Loading data')
    with open(args.annotation_file, 'r') as dataset_file:
        questions = []
        q_idx = []
        answers = []
        video_names = []
        ans_candidates = []
        for line in dataset_file:
            instance = json.loads(line)
            questions += [instance['q']]
            q_idx += [instance['qid']]
            if args.mode not in ['test']:
                answers += [instance['answer_idx']]
            else:
                answers += [0]
            video_names += [instance['vid_name']]
            ans_candidates += [[instance['a0'], instance['a1'], instance['a2'], instance['a3'], instance['a4']]]
    ans_candidates = np.asarray(ans_candidates)
    print(ans_candidates.shape)
    # ans_candidates: (num_ques, 5)
    print('number of questions: %s' % len(questions))
    # Either create the vocab or load it from disk
    if args.mode in ['train']:
        print('Building vocab')

        answer_token_to_idx = {'<UNK0>': 0, '<UNK1>': 1}
        question_answer_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
        for candidates in ans_candidates:
            for ans in candidates:
                ans = ans.lower()
                for token in nltk.word_tokenize(ans):
                    if token not in answer_token_to_idx:
                        answer_token_to_idx[token] = len(answer_token_to_idx)
                    if token not in question_answer_token_to_idx:
                        question_answer_token_to_idx[token] = len(question_answer_token_to_idx)
        print('Get answer_token_to_idx, num: %d' % len(answer_token_to_idx))

        question_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
        for i, q in enumerate(questions):
            question = q.lower()[:-1]
            for token in nltk.word_tokenize(question):
                if token not in question_token_to_idx:
                    question_token_to_idx[token] = len(question_token_to_idx)
                if token not in question_answer_token_to_idx:
                    question_answer_token_to_idx[token] = len(question_answer_token_to_idx)

        print('Get question_token_to_idx')
        print(len(question_token_to_idx))
        print('Get question_answer_token_to_idx')
        print(len(question_answer_token_to_idx))

        vocab = {
            'question_token_to_idx': question_token_to_idx,
            'answer_token_to_idx': answer_token_to_idx,
            'question_answer_token_to_idx': question_answer_token_to_idx,
        }

        print('Write into %s' % args.vocab_json.format(args.dataset, args.dataset))
        with open(args.vocab_json.format(args.dataset, args.dataset), 'w') as f:
            json.dump(vocab, f, indent=4)
    else:
        print('Loading vocab')
        with open(args.vocab_json.format(args.dataset, args.dataset), 'r') as f:
            vocab = json.load(f)
    multichoice_encoding_data(args, vocab, questions, video_names, q_idx, answers,
                                ans_candidates, mode=args.mode)
