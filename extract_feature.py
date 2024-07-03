import torch
import torch.nn as nn
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm

from torch.utils import data
from torchvision import transforms

from transformers import  Wav2Vec2Processor
from external.facebook.wav2vec2focctc import Wav2Vec2ForCTC
import librosa
import torchaudio
import soundfile as sf

import numpy as np
import pandas as pd
from PIL import Image

from decord import VideoReader
from decord import cpu

import argparse

def parse_arg():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # Param
    parser.add_argument('--data-dir', default="/public_bme/data/v-lijm/REACT_2024", type=str, help="dataset path")
    parser.add_argument('--save-dir', default="/public_bme/data/v-lijm/REACT_2024/features", type=str, help="the dir to save features")
    parser.add_argument('--split', type=str, help="split of dataset", choices=["train", "val", "test"], required=True)
    parser.add_argument('--type', type=str, help="type of features to extract", choices=["audio", "video"], required=True)
    
    args = parser.parse_args()
    return args


class Transform(object):
    def __init__(self, img_size=256, crop_size=224):
        self.img_size = img_size
        self.crop_size = crop_size
        
    def __call__(self, img):

        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            normalize
        ])

        img = transform(img)
        return img
  

def extract_audio_features(args):

    _list_path = pd.read_csv(os.path.join(args.data_dir, args.split + '.csv'), header=None, delimiter=',')
    _list_path = _list_path.drop(0)

    all_path = [path for path in list(_list_path.values[:, 1])] + [path for path in list(_list_path.values[:, 2])]

    for path in tqdm(all_path):
        audio_path = os.path.join(args.data_dir, args.split, 'Audio_files', path+'.wav')

        with torch.no_grad():
            '''----------------------------MFCC--------------------------------'''
            fps = 25
            n_frames = 751
            audio, sr = sf.read(audio_path)  # audio为声音数据, sr为声音采样率
            if audio.ndim == 2:
                audio = audio.mean(-1)
            frame_n_samples = int(sr / fps)
            curr_length = len(audio)
            target_length = frame_n_samples * n_frames
            if curr_length > target_length:
                audio = audio[:target_length]
            elif curr_length < target_length:
                audio = np.pad(audio, [0, target_length - curr_length])
            shifted_n_samples = 0
            curr_feats = []
            for i in range(n_frames):
                curr_samples = audio[i*frame_n_samples:shifted_n_samples + i*frame_n_samples + frame_n_samples] # [1764]
                curr_mfcc = torchaudio.compliance.kaldi.mfcc(torch.from_numpy(curr_samples).float().view(1, -1), sample_frequency=sr, use_energy=True, frame_length=40, frame_shift=40, num_ceps=26, num_mel_bins=26) # 16000HZ 采样率，40ms的窗口和步长 (25帧/s)
                curr_feats.append(curr_mfcc)

            curr_feats = torch.cat(curr_feats, dim=0) # [751, 26]

            '''----------------------------Wav2Vec2.0--------------------------------'''
            processor = Wav2Vec2Processor.from_pretrained("external/facebook/wav2vec2-base-960h")

            speech_array, sampling_rate = librosa.load(audio_path, sr=16000)## 重采样到16KHz
            # print(speech_array.shape) # 480253
            audio_clip = torch.FloatTensor(np.squeeze(processor(speech_array, sampling_rate=sampling_rate).input_values)).unsqueeze(0)
            # print(audio_clip.shape) #[1, 480253]

            audio_encoder = Wav2Vec2ForCTC.from_pretrained("external/facebook/wav2vec2-base-960h")
            audio_encoder.freeze_feature_extractor()

            curr_w2v2 = audio_encoder(audio_clip, frame_num=751).squeeze(0)
            s = curr_w2v2.shape[0]
            curr_w2v2 = curr_w2v2.reshape(s//2, -1)

            # print(curr_w2v2.shape) #([1500, 768])
            # return
            
            audio_features = [curr_feats, curr_w2v2]
            
        site, group, pid, clip = path.split('/')
        if not os.path.exists(os.path.join(args.save_dir, args.split, 'Audio_features', site, group, pid)):
            os.makedirs(os.path.join(args.save_dir, args.split, 'Audio_features', site, group, pid))

        torch.save(audio_features, os.path.join(args.save_dir, args.split, 'Audio_features', path+'.pth'))
    



def load_model(path_va, path_au, path_fer):
    model_va = torch.jit.load(path_va, map_location='cpu')
    model_va = model_va.cuda()
    model_va.eval()
    model_au = torch.jit.load(path_au, map_location='cpu')
    model_au = model_au.cuda()
    model_au.eval()
    model_fer = torch.jit.load(path_fer, map_location='cpu')
    model_fer = model_fer.cuda()
    model_fer.eval()
    return model_va, model_au, model_fer

def extract_video_features(args):
    _transform = Transform(img_size=256, crop_size=224)

    path_au = 'external/predictor/au/1/model.pt'
    path_va = 'external/predictor/va/1/model.pt'
    path_fer = 'external/predictor/fer/1/model.pt'
    models = load_model(path_va, path_au, path_fer)

    _list_path = pd.read_csv(os.path.join(args.data_dir, args.split + '.csv'), header=None, delimiter=',')
    _list_path = _list_path.drop(0)

    all_path = [path for path in list(_list_path.values[:, 1])] + [path for path in list(_list_path.values[:, 2])]

    total_length = 751

    pre_path = os.path.join(args.data_dir, 'test/Video_files')
    for path in tqdm(all_path):
        clip = []
        ab_video_path = os.path.join(pre_path, path+'.mp4')
        with open(ab_video_path, 'rb') as f:
            vr = VideoReader(f, ctx=cpu(0))
        for i in range(total_length):
            frame = vr[i]
            img=Image.fromarray(frame.asnumpy())
            img = _transform(img)
            clip.append(img.unsqueeze(0))

        video_clip = torch.cat(clip, dim=0).cuda()
        with torch.no_grad():
            va = models[0].regressor(models[0].encoder(models[0].regressor.avgpool, video_clip))  ## [s, 64]
            au = models[1].global_linear(models[1].backbone(video_clip)).flatten(1)  ## [s, 49, 512] -> [s, 25088]
            fer = models[2].global_pool(models[2].act2(models[2].bn2(models[2].
                conv_head(models[2].blocks(models[2].act1(models[2].bn1(models[2].conv_stem(video_clip)))))))) ## [s, 1408]
            video_features = [va, au, fer]
        site, group, pid, clip = path.split('/')
        if not os.path.exists(os.path.join(args.save_dir, args.split, 'Video_features', site, group, pid)):
            os.makedirs(os.path.join(args.save_dir, args.split, 'Video_features', site, group, pid))

        torch.save(video_features, os.path.join(args.save_dir, args.split, 'Video_features', path+'.pth')) 



def main(args):
    if args.type == 'video':
        extract_video_features(args)
    elif args.type == 'audio':
        extract_audio_features(args)

# ---------------------------------------------------------------------------------


if __name__=="__main__":
    args = parse_arg()
    os.environ["NUMEXPR_MAX_THREADS"] = '32'
    main(args)