import numpy as np
from torch.nn import functional as F

from .utils import *
from .CRN import CRN

class FeatureAggregation(nn.Module):
    def __init__(self, module_dim=512):
        super(FeatureAggregation, self).__init__()
        self.module_dim = module_dim

        self.q_proj = nn.Linear(module_dim, module_dim, bias=False)
        self.v_proj = nn.Linear(module_dim, module_dim, bias=False)

        self.cat = nn.Linear(2 * module_dim, module_dim)
        self.attn = nn.Linear(module_dim, 1)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.15)

    def forward(self, question_rep, visual_feat):
        visual_feat = self.dropout(visual_feat)
        q_proj = self.q_proj(question_rep)
        v_proj = self.v_proj(visual_feat)

        v_q_cat = torch.cat((v_proj, q_proj.unsqueeze(1) * v_proj), dim=-1)
        v_q_cat = self.cat(v_q_cat)
        v_q_cat = self.activation(v_q_cat)

        attn = self.attn(v_q_cat)  # (bz, k, 1)
        attn = F.softmax(attn, dim=1)  # (bz, k, 1)

        v_distill = (attn * visual_feat).sum(1)

        return v_distill

class PreSelection(nn.Module):
    def __init__(self, module_dim=512):
        super(PreSelection, self).__init__()
        self.module_dim = module_dim

        self.s_q_proj = nn.Linear(7 * module_dim, module_dim, bias=False)

        self.dropout = nn.Dropout(0.15)

    def forward(self, question_rep, subtitle_rep, ans_candidates_rep):
        subtitle_rep = self.dropout(subtitle_rep)

        batch_agg = np.reshape(
            np.tile(np.expand_dims(np.arange(question_rep.shape[0]), axis=1), [1, 5]), [-1])

        s_q_cat = torch.cat([subtitle_rep, question_rep * subtitle_rep, (ans_candidates_rep * subtitle_rep[batch_agg]).view((-1, self.module_dim * 5))], dim=-1)
        s_q_proj = self.s_q_proj(s_q_cat)

        return s_q_proj

class PreSelectionMulti(nn.Module):
    def __init__(self, module_dim=512):
        super(PreSelectionMulti, self).__init__()
        self.module_dim = module_dim

        self.s_q_proj = nn.Linear(7 * module_dim, module_dim, bias=False)

        self.dropout = nn.Dropout(0.15)

    def forward(self, question_rep, subsubtitle_rep, ans_candidates_rep):
        subsuvtitle_rep = self.dropout(subsubtitle_rep)

        batch_agg = np.reshape(
            np.tile(np.expand_dims(np.arange(question_rep.shape[0]), axis=1), [1, 5]), [-1])

        batch_agg_sub = np.reshape(
            np.tile(np.expand_dims(np.arange(question_rep.shape[0]), axis=1), [1, subsubtitle_rep.shape[1]]), [-1])

        batch_agg_sub_ans = np.reshape(
            np.tile(np.expand_dims(np.arange(ans_candidates_rep.shape[0]), axis=1), [1, subsubtitle_rep.shape[1]]), [-1])

        s_q_cat = torch.cat([subsubtitle_rep, question_rep[batch_agg_sub].view((-1, 6, self.module_dim)) * subsubtitle_rep, (ans_candidates_rep[batch_agg_sub_ans].view(-1, 6, self.module_dim) * subsubtitle_rep[batch_agg]).view((-1, 6, self.module_dim * 5))], dim=-1)
        s_q_proj = self.s_q_proj(s_q_cat)

        return s_q_proj


class InputUnitLinguistic(nn.Module):
    def __init__(self, vocab_size, wordvec_dim=300, rnn_dim=512, module_dim=512, bidirectional=True):
        super(InputUnitLinguistic, self).__init__()

        self.dim = module_dim

        self.bidirectional = bidirectional
        if bidirectional:
            rnn_dim = rnn_dim // 2

        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.tanh = nn.Tanh()
        self.encoder = nn.LSTM(wordvec_dim, rnn_dim, batch_first=True, bidirectional=bidirectional)
        self.embedding_dropout = nn.Dropout(p=0.15)
        self.question_dropout = nn.Dropout(p=0.18)

        self.module_dim = module_dim

    def forward(self, questions, question_len):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        questions_embedding = self.encoder_embed(questions)  # (batch_size, seq_len, dim_word)
        embed = self.tanh(self.embedding_dropout(questions_embedding))
        embed = nn.utils.rnn.pack_padded_sequence(embed, question_len.cpu(), batch_first=True,
                                                  enforce_sorted=False)

        self.encoder.flatten_parameters()
        _, (question_embedding, _) = self.encoder(embed)
        if self.bidirectional:
            question_embedding = torch.cat([question_embedding[0], question_embedding[1]], -1)
        question_embedding = self.question_dropout(question_embedding)

        return question_embedding


class InputUnitVisual(nn.Module):
    def __init__(self, k_max_frame_level, k_max_clip_level, spl_resolution, vision_dim, module_dim=512):
        super(InputUnitVisual, self).__init__()

        self.clip_level_motion_cond = CRN(module_dim, k_max_frame_level, k_max_frame_level, gating=False, spl_resolution=spl_resolution)
        self.clip_level_question_cond = CRN(module_dim, k_max_frame_level-2, k_max_frame_level-2, gating=True, spl_resolution=spl_resolution)
        self.video_level_motion_cond = CRN(module_dim, k_max_clip_level, k_max_clip_level, gating=False, spl_resolution=spl_resolution)
        self.video_level_question_cond = CRN(module_dim, k_max_clip_level-2, k_max_clip_level-2, gating=True, spl_resolution=spl_resolution)

        self.sequence_encoder = nn.LSTM(vision_dim, module_dim, batch_first=True, bidirectional=False)
        self.clip_level_motion_proj = nn.Linear(vision_dim, module_dim)
        self.video_level_motion_proj = nn.Linear(module_dim, module_dim)
        self.appearance_feat_proj = nn.Linear(vision_dim, module_dim)

        self.question_embedding_proj = nn.Linear(module_dim, module_dim)

        self.module_dim = module_dim
        self.activation = nn.ELU()

    def forward(self, appearance_video_feat, motion_video_feat, question_embedding):
        """
        Args:
            appearance_video_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            motion_video_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question_embedding: [Tensor] (batch_size, module_dim)
        return:
            encoded video feature: [Tensor] (batch_size, N, module_dim)
        """
        batch_size = appearance_video_feat.size(0)
        clip_level_crn_outputs = []
        question_embedding_proj = self.question_embedding_proj(question_embedding)
        for i in range(appearance_video_feat.size(1)):
            clip_level_motion = motion_video_feat[:, i, :]  # (bz, 2048)
            clip_level_motion_proj = self.clip_level_motion_proj(clip_level_motion)

            clip_level_appearance = appearance_video_feat[:, i, :, :]  # (bz, 16, 2048)
            clip_level_appearance_proj = self.appearance_feat_proj(clip_level_appearance)  # (bz, 16, 512)
            # clip level CRNs
            clip_level_crn_motion = self.clip_level_motion_cond(torch.unbind(clip_level_appearance_proj, dim=1),
                                                                clip_level_motion_proj)
            clip_level_crn_question = self.clip_level_question_cond(clip_level_crn_motion, question_embedding_proj)

            clip_level_crn_output = torch.cat(
                [frame_relation.unsqueeze(1) for frame_relation in clip_level_crn_question],
                dim=1)
            clip_level_crn_output = clip_level_crn_output.view(batch_size, -1, self.module_dim)
            clip_level_crn_outputs.append(clip_level_crn_output)

        # Encode video level motion
        _, (video_level_motion, _) = self.sequence_encoder(motion_video_feat)
        video_level_motion = video_level_motion.transpose(0, 1)
        video_level_motion_feat_proj = self.video_level_motion_proj(video_level_motion)
        # video level CRNs
        video_level_crn_motion = self.video_level_motion_cond(clip_level_crn_outputs, video_level_motion_feat_proj)
        video_level_crn_question = self.video_level_question_cond(video_level_crn_motion,
                                                                  question_embedding_proj.unsqueeze(1))

        video_level_crn_output = torch.cat([clip_relation.unsqueeze(1) for clip_relation in video_level_crn_question],
                                           dim=1)
        video_level_crn_output = video_level_crn_output.view(batch_size, -1, self.module_dim)

        return video_level_crn_output

class InputUnitVisualNoMotion(nn.Module):
    def __init__(self, k_max_frame_level, k_max_clip_level, spl_resolution, vision_dim, module_dim=512):
        super(InputUnitVisualNoMotion, self).__init__()

        self.clip_level_question_cond = CRN(module_dim, k_max_frame_level, k_max_frame_level, gating=True, spl_resolution=spl_resolution)
        self.video_level_question_cond = CRN(module_dim, k_max_clip_level, k_max_clip_level, gating=True, spl_resolution=spl_resolution)

        self.sequence_encoder = nn.LSTM(vision_dim, module_dim, batch_first=True, bidirectional=False)
        self.clip_level_motion_proj = nn.Linear(vision_dim, module_dim)
        self.video_level_motion_proj = nn.Linear(module_dim, module_dim)
        self.appearance_feat_proj = nn.Linear(vision_dim, module_dim)

        self.question_embedding_proj = nn.Linear(module_dim, module_dim)

        self.module_dim = module_dim
        self.activation = nn.ELU()

    def forward(self, appearance_video_feat, question_embedding):
        """
        Args:
            appearance_video_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            question_embedding: [Tensor] (batch_size, module_dim)
        return:
            encoded video feature: [Tensor] (batch_size, N, module_dim)
        """
        batch_size = appearance_video_feat.size(0)
        clip_level_crn_outputs = []
        question_embedding_proj = self.question_embedding_proj(question_embedding)
        for i in range(appearance_video_feat.size(1)):
            clip_level_appearance = appearance_video_feat[:, i, :, :]  # (bz, 16, 2048)
            clip_level_appearance_proj = self.appearance_feat_proj(clip_level_appearance)  # (bz, 16, 512)
            # clip level CRNs
            clip_level_crn_motion = torch.unbind(clip_level_appearance_proj, dim=1)
            clip_level_crn_question = self.clip_level_question_cond(clip_level_crn_motion, question_embedding_proj)

            clip_level_crn_output = torch.cat(
                [frame_relation.unsqueeze(1) for frame_relation in clip_level_crn_question],
                dim=1)
            clip_level_crn_output = clip_level_crn_output.view(batch_size, -1, self.module_dim)
            clip_level_crn_outputs.append(clip_level_crn_output)

        # Encode video level motion
        video_level_crn_question = self.video_level_question_cond(clip_level_crn_outputs,
                                                                  question_embedding_proj.unsqueeze(1))

        video_level_crn_output = torch.cat([clip_relation.unsqueeze(1) for clip_relation in video_level_crn_question],
                                           dim=1)
        video_level_crn_output = video_level_crn_output.view(batch_size, -1, self.module_dim)

        return video_level_crn_output


class InputUnitTextual(nn.Module):
    def __init__(self, vocab_size, spl_resolution, wordvec_dim=300, rnn_dim=512, module_dim=512, bidirectional=True):
        super(InputUnitTextual, self).__init__()

        self.dim = module_dim

        self.bidirectional = bidirectional
        if bidirectional:
            rnn_dim = rnn_dim // 2

        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.tanh = nn.Tanh()
        self.encoder = nn.LSTM(wordvec_dim, rnn_dim, batch_first=True, bidirectional=bidirectional)
        self.embedding_dropout = nn.Dropout(p=0.15)
        self.subtitle_dropout = nn.Dropout(p=0.18)

        self.pre_selection = PreSelection(module_dim)
        self.pre_selection_multi = PreSelectionMulti(module_dim)
        self.crn_subtitle = CRN(module_dim, 6, 6, gating=True, spl_resolution=spl_resolution)

        self.module_dim = module_dim

    def forward(self, subtitle, subtitle_len, subsubtitle, subsubtitle_len, question_embedding, ans_candidates_embedding):
        """
        Args:
            subtitle: [Tensor] (batch_size, max_subtitle_length)
            subtitle_len: [Tensor] (batch_size)
            subsubtitle: [Tensor] (batch_size, M, max_subdivision_length)
            subsubtitle_len [Tensor] (batch_size)
        return:
            tuple:
            subtitle representation [Tensor] (batch_size, module_dim)
            subsubtitles representation [Tensor] (batch_size, M, module_dim)
        """
        subtitle_embedding = self.encoder_embed(subtitle)  # (batch_size, seq_len, dim_word)
        embed = self.tanh(self.embedding_dropout(subtitle_embedding))
        embed = nn.utils.rnn.pack_padded_sequence(embed, subtitle_len.cpu(), batch_first=True,
                                                  enforce_sorted=False)

        self.encoder.flatten_parameters()
        _, (subtitle_embedding, _) = self.encoder(embed)
        if self.bidirectional:
            subtitle_embedding = torch.cat([subtitle_embedding[0], subtitle_embedding[1]], -1)
        subtitle_embedding = self.subtitle_dropout(subtitle_embedding)

        # pass with every sub-subtitle:
        subsubtitle_embedding = torch.zeros((subsubtitle.shape[1], subtitle.shape[0], subtitle_embedding.shape[1]))
        if subtitle.is_cuda:
            subsubtitle_embedding = subsubtitle_embedding.to(subtitle.get_device())
        subsubtitle = torch.transpose(subsubtitle, 0, 1)
        for i in range(len(subsubtitle)):
            subsub_embed = self.encoder_embed(subsubtitle[i])  # (batch_size, seq_len, dim_word)
            embed = self.tanh(self.embedding_dropout(subsub_embed))
            embed = nn.utils.rnn.pack_padded_sequence(embed, subsubtitle_len.cpu(), batch_first=True,
                                                        enforce_sorted=False)

            self.encoder.flatten_parameters()
            _, (subsub_embed, _) = self.encoder(embed)
            if self.bidirectional:
                subsub_embed = torch.cat([subsub_embed[0], subsub_embed[1]], -1)
            subsub_embed = self.subtitle_dropout(subsub_embed)
            subsubtitle_embedding[i] = subsub_embed

        subsubtile = torch.transpose(subsubtitle, 0, 1)
        subsubtitle_embedding = torch.transpose(subsubtitle_embedding, 0, 1)

        # Pre-Selection
        subtitle_embedding = self.pre_selection(question_embedding, subtitle_embedding, ans_candidates_embedding)
        subsubtitle_embedding = self.pre_selection_multi(question_embedding, subsubtitle_embedding, ans_candidates_embedding)

        crn_output = self.crn_subtitle(torch.unbind(subsubtitle_embedding, dim=1), subtitle_embedding)

        crn_output = torch.max(torch.stack(crn_output), dim=0)[0]

        return crn_output
        # return subtitle_embedding, None


class OutputUnitOpenEnded(nn.Module):
    def __init__(self, module_dim=512, num_answers=1000):
        super(OutputUnitOpenEnded, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim)

        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 2, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, num_answers))

    def forward(self, question_embedding, visual_embedding):
        question_embedding = self.question_proj(question_embedding)
        out = torch.cat([visual_embedding, question_embedding], 1)
        out = self.classifier(out)

        return out


class OutputUnitMultiChoices(nn.Module):
    def __init__(self, module_dim=512):
        super(OutputUnitMultiChoices, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim)

        self.ans_candidates_proj = nn.Linear(module_dim, module_dim)

        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 4, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, 1))

    def forward(self, question_embedding, q_visual_embedding, ans_candidates_embedding,
                a_visual_embedding):
        question_embedding = self.question_proj(question_embedding)
        ans_candidates_embedding = self.ans_candidates_proj(ans_candidates_embedding)
        out = torch.cat([q_visual_embedding, question_embedding, a_visual_embedding,
                         ans_candidates_embedding], 1)
        out = self.classifier(out)

        return out


class OutputUnitMultiChoicesSubtitle(nn.Module):
    def __init__(self, module_dim=512):
        super(OutputUnitMultiChoicesSubtitle, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim)

        self.ans_candidates_proj = nn.Linear(module_dim, module_dim)

        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 5, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, 1))

    def forward(self, question_embedding, q_visual_embedding, ans_candidates_embedding,
                a_visual_embedding, s_embedding):
        question_embedding = self.question_proj(question_embedding)
        ans_candidates_embedding = self.ans_candidates_proj(ans_candidates_embedding)
        out = torch.cat([q_visual_embedding, question_embedding, a_visual_embedding,
                         ans_candidates_embedding, s_embedding], 1)
        out = self.classifier(out)

        return out


class OutputUnitCount(nn.Module):
    def __init__(self, module_dim=512):
        super(OutputUnitCount, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim)

        self.regression = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 2, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, 1))

    def forward(self, question_embedding, visual_embedding):
        question_embedding = self.question_proj(question_embedding)
        out = torch.cat([visual_embedding, question_embedding], 1)
        out = self.regression(out)

        return out


class HCRNNetwork(nn.Module):
    def __init__(self, vision_dim, module_dim, word_dim, k_max_frame_level, k_max_clip_level, spl_resolution, vocab, question_type):
        super(HCRNNetwork, self).__init__()

        self.question_type = question_type
        self.feature_aggregation = FeatureAggregation(module_dim)

        if self.question_type in ['action', 'transition']:
            encoder_vocab_size = len(vocab['question_answer_token_to_idx'])
            self.linguistic_input_unit = InputUnitLinguistic(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,
                                                             module_dim=module_dim, rnn_dim=module_dim)
            self.visual_input_unit = InputUnitVisual(k_max_frame_level=k_max_frame_level, k_max_clip_level=k_max_clip_level, spl_resolution=spl_resolution, vision_dim=vision_dim, module_dim=module_dim)
            self.output_unit = OutputUnitMultiChoices(module_dim=module_dim)

        elif self.question_type == 'tvqa':
            encoder_vocab_size = len(vocab['question_answer_token_to_idx'])
            encoder_subtitle_vocab_size = len(vocab['subtitle_token_to_idx'])
            self.linguistic_input_unit = InputUnitLinguistic(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,
                                                             module_dim=module_dim, rnn_dim=module_dim)
            self.textual_input_unit = InputUnitTextual(vocab_size=encoder_subtitle_vocab_size, wordvec_dim=word_dim, spl_resolution=spl_resolution,
                                                             module_dim=module_dim, rnn_dim=module_dim)
            self.visual_input_unit = InputUnitVisualNoMotion(k_max_frame_level=k_max_frame_level, k_max_clip_level=k_max_clip_level, spl_resolution=spl_resolution, vision_dim=vision_dim, module_dim=module_dim)
            self.output_unit = OutputUnitMultiChoicesSubtitle(module_dim=module_dim)

        elif self.question_type == 'count':
            encoder_vocab_size = len(vocab['question_token_to_idx'])
            self.linguistic_input_unit = InputUnitLinguistic(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,
                                                             module_dim=module_dim, rnn_dim=module_dim)
            self.visual_input_unit = InputUnitVisual(k_max_frame_level=k_max_frame_level, k_max_clip_level=k_max_clip_level, spl_resolution=spl_resolution, vision_dim=vision_dim, module_dim=module_dim)
            self.output_unit = OutputUnitCount(module_dim=module_dim)
        else:
            encoder_vocab_size = len(vocab['question_token_to_idx'])
            self.num_classes = len(vocab['answer_token_to_idx'])
            self.linguistic_input_unit = InputUnitLinguistic(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,
                                                             module_dim=module_dim, rnn_dim=module_dim)
            self.visual_input_unit = InputUnitVisual(k_max_frame_level=k_max_frame_level, k_max_clip_level=k_max_clip_level, spl_resolution=spl_resolution, vision_dim=vision_dim, module_dim=module_dim)
            self.output_unit = OutputUnitOpenEnded(num_answers=self.num_classes)

        init_modules(self.modules(), w_init="xavier_uniform")
        #nn.init.uniform_(self.linguistic_input_unit.encoder_embed.weight, -1.0, 1.0)

    def forward(self, ans_candidates, ans_candidates_len, video_appearance_feat, video_motion_feat, question,
                question_len, subtitle, subtitle_len, subsubtitle, subsubtitle_len):
        """
        Args:
            ans_candidates: [Tensor] (batch_size, 5, max_ans_candidates_length)
            ans_candidates_len: [Tensor] (batch_size, 5)
            video_appearance_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            video_motion_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
            subtitle: [Tensor] (batch_size, max_subtitle_length) (max_subtitle_length <= 256)
            subtitle_len: [Tensor] (batch_size)
            subsubtitle: [Tensor] (batch_size, M, max_subdivision_length)
            subsubtitle_len [Tensor] (batch_size)
        return:
            logits.
        """
        batch_size = question.size(0)
        if self.question_type in ['frameqa', 'count', 'none']:
            # get image, word, and sentence embeddings
            question_embedding = self.linguistic_input_unit(questions, question_len)
            visual_embedding = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding)

            visual_embedding = self.feature_aggregation(question_embedding, visual_embedding)

            out = self.output_unit(question_embedding, visual_embedding)
        elif self.question_type == 'tvqa':
            question_embedding = self.linguistic_input_unit(question, question_len)

            visual_embedding = self.visual_input_unit(video_appearance_feat, question_embedding)

            q_visual_embedding = self.feature_aggregation(question_embedding, visual_embedding)

            # ans_candidates: (batch_size, num_choices, max_len)
            ans_candidates_agg = ans_candidates.view(-1, ans_candidates.size(2))
            ans_candidates_len_agg = ans_candidates_len.view(-1)

            batch_agg = np.reshape(
                np.tile(np.expand_dims(np.arange(batch_size), axis=1), [1, 5]), [-1])

            ans_candidates_embedding = self.linguistic_input_unit(ans_candidates_agg, ans_candidates_len_agg)

            s_embedding = self.textual_input_unit(subtitle, subtitle_len, subsubtitle, subsubtitle_len, question_embedding, ans_candidates_embedding)

            a_visual_embedding = self.feature_aggregation(ans_candidates_embedding, visual_embedding[batch_agg])
            out = self.output_unit(question_embedding[batch_agg], q_visual_embedding[batch_agg],
                                   ans_candidates_embedding,
                                   a_visual_embedding, s_embedding[batch_agg])
        else:
            question_embedding = self.linguistic_input_unit(question, question_len)
            visual_embedding = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding)

            q_visual_embedding = self.feature_aggregation(question_embedding, visual_embedding)

            # ans_candidates: (batch_size, num_choices, max_len)
            ans_candidates_agg = ans_candidates.view(-1, ans_candidates.size(2))
            ans_candidates_len_agg = ans_candidates_len.view(-1)

            batch_agg = np.reshape(
                np.tile(np.expand_dims(np.arange(batch_size), axis=1), [1, 5]), [-1])

            ans_candidates_embedding = self.linguistic_input_unit(ans_candidates_agg, ans_candidates_len_agg)

            a_visual_embedding = self.feature_aggregation(ans_candidates_embedding, visual_embedding[batch_agg])
            out = self.output_unit(question_embedding[batch_agg], q_visual_embedding[batch_agg],
                                   ans_candidates_embedding,
                                   a_visual_embedding)
        return out
