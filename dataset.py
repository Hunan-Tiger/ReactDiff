import os
import torch
from torch.utils import data
from torchvision import transforms
import copy
import numpy as np
import pickle
from tqdm import tqdm
import random,math
import time
import pandas as pd
from PIL import Image
import cv2
from torch.utils.data import DataLoader
from multiprocessing import Pool
import torchaudio
from scipy.io import loadmat
torchaudio.set_audio_backend("sox_io")
from functools import cmp_to_key, partial


def subtract_mean_face(e, mean_face):
   return e - mean_face
class ReactionDataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, root_path, split, img_size=256, crop_size=224, clip_length=751, fps=25,
                 load_audio_s=False, load_audio_l=False, load_video_s=False, load_video_l=False, load_emotion_s=False, load_emotion_l=False, load_3dmm_s=False, load_3dmm_l=False,
                 load_ref=True, repeat_mirrored=True):
        """
        Args:
            root_path: (str) Path to the data folder.
            split: (str) 'train' or 'val' or 'test' split.
            img_size: (int) Size of the image.
            crop_size: (int) Size of the crop.
            clip_length: (int) Number of frames in a clip.
            fps: (int) Frame rate of the video.
            load_audio: (bool) Whether to load audio features.
            load_video_s: (bool) Whether to load speaker video features.
            load_video_l: (bool) Whether to load listener video features.
            load_emotion: (bool) Whether to load emotion labels.
            load_3dmm: (bool) Whether to load 3DMM parameters.
            repeat_mirrored: (bool) Whether to extend dataset with mirrored speaker/listener. This is used for val/test.
        """

        self._root_path = root_path    ## "./REACT_2024"
        self._clip_length = clip_length 
        self._fps = fps
        self._split = split

        self._data_path = os.path.join(self._root_path, self._split)
        ##      react_2024/train  or  react_2024/val
        self._list_path = pd.read_csv(os.path.join(self._root_path, self._split + '.csv'), header=None, delimiter=',')
        ##      REACT_2024/train.csv
        self._list_path = self._list_path.drop(0)  ## 删掉第一列（全是序号）

        self.load_audio_s = load_audio_s
        self.load_audio_l = load_audio_l
        self.load_video_s = load_video_s
        self.load_video_l = load_video_l

        self.load_3dmm_s = load_3dmm_s
        self.load_3dmm_l = load_3dmm_l
        self.load_emotion_s = load_emotion_s
        self.load_emotion_l = load_emotion_l

        self.load_ref = load_ref

        self._features_va_path = os.path.join(self._root_path, 'features', self._split)
        self._audio_path = os.path.join(self._features_va_path, 'Audio_features')     ## react_2024/train/Audio_files
        self._video_path = os.path.join(self._features_va_path, 'Video_features')    ##  设置video文件路径

        
        self._emotion_path = os.path.join(self._data_path, 'Emotion')
        self._3dmm_path = os.path.join(self._data_path, '3D_FV_files')

        self.mean_face = torch.FloatTensor(
            np.load('external/FaceVerse/mean_face.npy').astype(np.float32)).view(1, 1, -1)  ## 在列方向展开
        self.std_face = torch.FloatTensor(
            np.load('external/FaceVerse/std_face.npy').astype(np.float32)).view(1, 1, -1)

        self._transform_3dmm = transforms.Lambda(partial(subtract_mean_face, mean_face=self.mean_face))

        speaker_path = list(self._list_path.values[:, 1])  ## NoXI/065_2016-04-14_Nottingham/Expert_video/1, ...
        listener_path = list(self._list_path.values[:, 2]) ## NoXI/065_2016-04-14_Nottingham/Novice_video/1, ...

        if self._split in ["val", "test"] or repeat_mirrored:  # training is always mirrored as data augmentation
            speaker_path_tmp = speaker_path + listener_path
            listener_path_tmp = listener_path + speaker_path
            speaker_path = speaker_path_tmp
            listener_path = listener_path_tmp  ## 说话者和听者数据扩增一倍

        self.data_list = []
        ## sp ： NoXI/065_2016-04-14_Nottingham/Expert_video（Novice_video）/1
        ## lp :  NoXI/065_2016-04-14_Nottingham/Novice_video/1
        for i, (sp, lp) in enumerate(zip(speaker_path, listener_path)):
            ab_speaker_video_path = os.path.join(self._video_path, sp + '.pth')
            ab_speaker_audio_path = os.path.join(self._audio_path, sp + '.pth')
            tsp = sp.replace('Expert_video', 'P1') if 'Expert_video' in sp else sp.replace('Novice_video', 'P2')
            ab_speaker_emotion_path = os.path.join(self._emotion_path, tsp + '.csv') ## Expert_video需替换为P1
            ab_speaker_3dmm_path = os.path.join(self._3dmm_path, sp + '.npy')

            ab_listener_video_path = os.path.join(self._video_path, lp + '.pth')
            ab_listener_audio_path = os.path.join(self._audio_path, lp + '.pth')
            tlp = lp.replace('Expert_video', 'P1') if 'Expert_video' in lp else lp.replace('Novice_video', 'P2')
            ab_listener_emotion_path = os.path.join(self._emotion_path, tlp + '.csv')
            ab_listener_3dmm_path = os.path.join(self._3dmm_path, lp + '.npy')

            self.data_list.append(
                {'speaker_video_path': ab_speaker_video_path, 'speaker_audio_path': ab_speaker_audio_path,
                 'speaker_emotion_path': ab_speaker_emotion_path, 'speaker_3dmm_path': ab_speaker_3dmm_path,
                 'listener_video_path': ab_listener_video_path, 'listener_audio_path': ab_listener_audio_path,
                 'listener_emotion_path': ab_listener_emotion_path, 'listener_3dmm_path': ab_listener_3dmm_path})

        self._len = len(self.data_list)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        data = self.data_list[index]

        # ========================= Data Augmentation ==========================
        changed_sign = 0
        if self._split == 'train':  # only done at training time
            changed_sign = random.randint(0, 1)

        speaker_prefix = 'speaker' #if changed_sign == 0 else 'listener'
        listener_prefix = 'listener' #if changed_sign == 0 else 'speaker'


        # ========================= Load Speaker & Listener video clip ==========================
        speaker_video_path = data[f'{speaker_prefix}_video_path']
        listener_video_path = data[f'{listener_prefix}_video_path']

        total_length = 751
        cp = random.randint(0, total_length - 1 - self._clip_length) if self._clip_length < total_length else 0

        speaker_video_clip, listener_video_clip = 0, 0
        if self.load_video_s:
            # 原说话者视频
            # video = np.load(speaker_video_path)  ## 加载npy文件, [751, 3, 224, 224]
            # speaker_video_clip = video[cp: cp + self._clip_length, :, :, :]
            videos = torch.load(speaker_video_path, map_location='cpu')
            speaker_video_clip_pre = torch.cat([videos[0][cp: cp + self._clip_length], videos[1][cp: cp + self._clip_length]],dim = -1)
            speaker_video_clip = torch.cat([speaker_video_clip_pre, videos[2][cp: cp + self._clip_length]], dim = -1)

        # listener video clip only needed for val/test
        if self.load_video_l:
            # 原听者视频
            # video = np.load(listener_video_path)  ## 加载npy文件
            # listener_video_clip = video[cp: cp + self._clip_length, :, :, :]
            videos = torch.load(listener_video_path, map_location='cpu')
            listener_video_clip_pre = torch.cat([videos[0][cp: cp + self._clip_length], videos[1][cp: cp + self._clip_length]],dim = -1)
            listener_video_clip = torch.cat([listener_video_clip_pre, videos[2][cp: cp + self._clip_length]], dim = -1)
            # [256, 64+25088+1408=26560]

        # ========================= Load Listener Reference ==========================
        listener_reference, listener_video_data, speaker_video_data = 0, 0, 0

        if self.load_ref:
            _, _, _, _, _, _, _, _, site, group, pid, clip = listener_video_path.split('/')
            # listener_video_path = /public_bme/data/v-lijm/REACT_2024/features/train/Video_features/NoXI/007_2016-03-21_Paris/Expert_video/1.pth
            listener_video_ = os.path.join(self._root_path, 'video_data', site, group, pid, clip[:-4]+'.npy')
            listener_video = np.load(listener_video_)
            listener_reference = listener_video[0]
            listener_video_data = listener_video[cp: cp + self._clip_length]

            _, _, _, _, _, _, _, _, site, group, pid, clip = speaker_video_path.split('/')
            speaker_video_ = os.path.join(self._root_path, 'video_data', site, group, pid, clip[:-4]+'.npy')
            speaker_video = np.load(speaker_video_)
            speaker_video_data = speaker_video[cp: cp + self._clip_length]

        # ========================= Load Speaker audio clip ==========================
        listener_audio_clip, speaker_audio_clip = 0, 0
        if self.load_audio_s:
          speaker_audio_path = data[f'{speaker_prefix}_audio_path']
          audios = torch.load(speaker_audio_path, map_location='cpu')
          speaker_audio_clip = torch.cat([audios[0][cp: cp + self._clip_length], audios[1][cp: cp + self._clip_length]], dim=-1) #[256, 26+1536]
        
        if self.load_audio_l:
          listener_audio_path = data[f'{listener_prefix}_audio_path']
          audios = torch.load(listener_audio_path, map_location='cpu')
          listener_audio_clip = torch.cat([audios[0][cp: cp + self._clip_length], audios[1][cp: cp + self._clip_length]], dim=-1)

        # ========================= Load Speaker & Listener emotion ==========================
        listener_emotion, speaker_emotion = 0, 0
        if self.load_emotion_l:
            listener_emotion_path = data[f'{listener_prefix}_emotion_path']
            listener_emotion = pd.read_csv(listener_emotion_path, header=None, delimiter=',')
            listener_emotion = torch.from_numpy(np.array(listener_emotion.drop(0)).astype(np.float32))[
                               cp: cp + self._clip_length]

        if self.load_emotion_s:
            speaker_emotion_path = data[f'{speaker_prefix}_emotion_path']
            speaker_emotion = pd.read_csv(speaker_emotion_path, header=None, delimiter=',')
            speaker_emotion = torch.from_numpy(np.array(speaker_emotion.drop(0)).astype(np.float32))[
                              cp: cp + self._clip_length]


        # ========================= Load Listener 3DMM ==========================
        listener_3dmm = 0
        if self.load_3dmm_l:
            listener_3dmm_path = data[f'{listener_prefix}_3dmm_path']
            listener_3dmm = torch.FloatTensor(np.load(listener_3dmm_path)).squeeze()
            listener_3dmm = listener_3dmm[cp: cp + self._clip_length]
            listener_3dmm = self._transform_3dmm(listener_3dmm)[0]

        speaker_3dmm = 0
        if self.load_3dmm_s:
            speaker_3dmm_path = data[f'{speaker_prefix}_3dmm_path']
            speaker_3dmm = torch.FloatTensor(np.load(speaker_3dmm_path)).squeeze()
            speaker_3dmm = speaker_3dmm[cp: cp + self._clip_length]
            speaker_3dmm = self._transform_3dmm(speaker_3dmm)[0]
            

        return speaker_video_clip, listener_video_clip, speaker_audio_clip, listener_audio_clip, speaker_emotion, speaker_3dmm, listener_emotion, listener_3dmm, listener_reference, speaker_video_data, listener_video_data

    def __len__(self):
        return self._len


def get_dataloader(conf, split, 
                   load_audio_s=False, load_audio_l=False, load_video_s=False, load_video_l=False, load_emotion_s=False, load_emotion_l=False, load_3dmm_s=False, load_3dmm_l=False,
                   load_ref=False, repeat_mirrored=True):
    assert split in ["train", "val", "test"], "split must be in [train, val, test]"
    dataset = ReactionDataset(conf.dataset_path, split, img_size=conf.img_size, crop_size=conf.crop_size,
                              clip_length=conf.clip_length, fps=25,
                              load_audio_s=load_audio_s, load_audio_l=load_audio_l, load_video_s=load_video_s, load_video_l=load_video_l,
                              load_emotion_s=load_emotion_s, load_emotion_l=load_emotion_l, load_3dmm_s=load_3dmm_s,
                              load_3dmm_l=load_3dmm_l, load_ref=load_ref, repeat_mirrored=repeat_mirrored)
    shuffle = True if split == "train" else False
    dataloader = DataLoader(dataset=dataset, batch_size=conf.batch_size, shuffle=shuffle, num_workers=conf.num_workers)
    return dataloader
