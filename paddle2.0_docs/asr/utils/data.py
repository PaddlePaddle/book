import json
import wave

import librosa
import numpy as np
import soundfile
from paddle.io import Dataset


# 加载二进制音频文件，转成短时傅里叶变换
def load_audio_stft(wav_path, mean=None, std=None):
    with wave.open(wav_path) as wav:
        wav = np.frombuffer(
            wav.readframes(wav.getnframes()), dtype="int16").astype("float32")
    stft = librosa.stft(
        wav, n_fft=255, hop_length=160, win_length=200, window="hamming")
    spec, phase = librosa.magphase(stft)
    spec = np.log1p(spec)
    if mean is not None and std is not None:
        spec = (spec - mean) / std
    return spec


# 读取音频文件转成梅尔频率倒谱系数(MFCCs)
def load_audio_mfcc(wav_path, mean=None, std=None):
    wav, sr = librosa.load(wav_path, sr=16000)
    mfccs = librosa.feature.mfcc(
        y=wav, sr=sr, n_mfcc=128, n_fft=512, hop_length=128).astype("float32")
    if mean is not None and std is not None:
        mfccs = (mfccs - mean) / std
    return mfccs


# 改变音频采样率为16000Hz
def change_rate(audio_path):
    data, sr = soundfile.read(audio_path)
    if sr != 16000:
        data = librosa.resample(data, sr, target_sr=16000)
        soundfile.write(audio_path, data, samplerate=16000)


# 音频数据加载器
class PPASRDataset(Dataset):
    def __init__(self,
                 data_list,
                 dict_path,
                 mean=None,
                 std=None,
                 min_duration=0,
                 max_duration=-1):
        super(PPASRDataset, self).__init__()
        self.mean = mean
        self.std = std
        # 获取数据列表
        with open(data_list, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.data_list = []
        for line in lines:
            line = json.loads(line)
            # 跳过超出长度限制的音频
            if line["duration"] < min_duration:
                continue
            if max_duration != -1 and line["duration"] > max_duration:
                continue
            self.data_list.append([line["audio_path"], line["text"]])
        # 加载数据字典
        with open(dict_path, 'r', encoding='utf-8') as f:
            labels = eval(f.read())
        self.vocabulary = dict([(labels[i], i) for i in range(len(labels))])

    def __getitem__(self, idx):
        # 分割音频路径和标签
        wav_path, transcript = self.data_list[idx]
        # 读取音频并转换为梅尔频率倒谱系数(MFCCs)
        mfccs = load_audio_mfcc(wav_path, self.mean, self.std)
        # 将字符标签转换为int数据
        transcript = list(
            filter(None, [self.vocabulary.get(x) for x in transcript]))
        transcript = np.array(transcript, dtype='int32')
        return mfccs, transcript

    def __len__(self):
        return len(self.data_list)


# 对一个batch的数据处理
def collate_fn(batch):
    # 找出音频长度最长的
    batch = sorted(batch, key=lambda sample: sample[0].shape[1], reverse=True)
    freq_size = batch[0][0].shape[0]
    max_audio_length = batch[0][0].shape[1]
    batch_size = len(batch)
    # 找出标签最长的
    batch_temp = sorted(batch, key=lambda sample: len(sample[1]), reverse=True)
    max_label_length = len(batch_temp[0][1])
    # 以最大的长度创建0张量
    inputs = np.zeros(
        (batch_size, freq_size, max_audio_length), dtype='float32')
    labels = np.zeros((batch_size, max_label_length), dtype='int32')
    input_lens = []
    label_lens = []
    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.shape[1]
        label_length = target.shape[0]
        # 将数据插入都0张量中，实现了padding
        inputs[x, :, :seq_length] = tensor[:, :]
        labels[x, :label_length] = target[:]
        input_lens.append(seq_length)
        label_lens.append(len(target))
    input_lens = np.array(input_lens, dtype='int64')
    label_lens = np.array(label_lens, dtype='int64')
    return inputs, labels, input_lens, label_lens
