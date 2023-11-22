import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import time
from text import text_to_sequence

import os


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)

        return txt


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_mask_from_lengths(lengths, max_len=None):
    if max_len == None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()

    return mask


def get_WaveGlow(waveglow_path):
    waveglow_path = os.path.join(waveglow_path, "waveglow_256channels.pt")
    wave_glow = torch.load(waveglow_path)['model']
    wave_glow = wave_glow.remove_weightnorm(wave_glow)
    wave_glow.cuda().eval()
    for m in wave_glow.modules():
        if 'Conv' in str(type(m)):
            setattr(m, 'padding_mode', 'zeros')

    return wave_glow


def pad_1D(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = np.pad(x, (0, length - x.shape[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_1D_tensor(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = F.pad(x, (0, length - x.shape[0]))
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = torch.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):

    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad_2D_tensor(inputs, maxlen=None):

    def pad(x, max_len):
        if x.size(0) > max_len:
            raise ValueError("not max_len")

        s = x.size(1)
        x_padded = F.pad(x, (0, 0, 0, max_len-x.size(0)))
        return x_padded[:, :s]

    if maxlen:
        output = torch.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(x.size(0) for x in inputs)
        output = torch.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        out_list = list()
        max_len = mel_max_length
        for i, batch in enumerate(input_ele):
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len-batch.size(0)), "constant", 0.0)
            out_list.append(one_batch_padded)
        out_padded = torch.stack(out_list)
        return out_padded
    else:
        out_list = list()
        max_len = max([input_ele[i].size(0)for i in range(len(input_ele))])

        for i, batch in enumerate(input_ele):
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len-batch.size(0)), "constant", 0.0)
            out_list.append(one_batch_padded)
        out_padded = torch.stack(out_list)
        return out_padded
    
def get_data_to_buffer(train_config):
    buffer = list()
    text = process_text(train_config.data_path)

    start = time.perf_counter()
    for i in tqdm(range(len(text))):

        mel_gt_name = os.path.join(
            train_config.mel_ground_truth, "ljspeech-mel-%05d.npy" % (i+1))
        mel_gt_target = np.load(mel_gt_name)
        duration = np.load(os.path.join(
            train_config.alignment_path, str(i)+".npy"))
        character = text[i][0:len(text[i])-1]
        character = np.array(
            text_to_sequence(character, train_config.text_cleaners))

        character = torch.from_numpy(character)
        duration = torch.from_numpy(duration)
        mel_gt_target = torch.from_numpy(mel_gt_target)

        buffer.append({"text": character, "duration": duration,
                       "mel_target": mel_gt_target})

    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end-start))

    return buffer

def reprocess_tensor(batch, cut_list):
    texts = [batch[ind]["text"] for ind in cut_list]
    mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
    durations = [batch[ind]["duration"] for ind in cut_list]

    length_text = np.array([])
    for text in texts:
        length_text = np.append(length_text, text.size(0))

    src_pos = list()
    max_len = int(max(length_text))
    for length_src_row in length_text:
        src_pos.append(np.pad([i+1 for i in range(int(length_src_row))],
                              (0, max_len-int(length_src_row)), 'constant'))
    src_pos = torch.from_numpy(np.array(src_pos))

    length_mel = np.array(list())
    for mel in mel_targets:
        length_mel = np.append(length_mel, mel.size(0))

    mel_pos = list()
    max_mel_len = int(max(length_mel))
    for length_mel_row in length_mel:
        mel_pos.append(np.pad([i+1 for i in range(int(length_mel_row))],
                              (0, max_mel_len-int(length_mel_row)), 'constant'))
    mel_pos = torch.from_numpy(np.array(mel_pos))

    texts = pad_1D_tensor(texts)
    durations = pad_1D_tensor(durations)
    mel_targets = pad_2D_tensor(mel_targets)

    out = {"text": texts,
           "mel_target": mel_targets,
           "duration": durations,
           "mel_pos": mel_pos,
           "src_pos": src_pos,
           "mel_max_len": max_mel_len}

    return out

def create_alignment(base_mat, duration_predictor_output):
    N, L = duration_predictor_output.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(duration_predictor_output[i][j]):
                base_mat[i][count+k][j] = 1
            count = count + duration_predictor_output[i][j]
    return base_mat

