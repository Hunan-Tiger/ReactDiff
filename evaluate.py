import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from tqdm import tqdm
import logging
from dataset import ReactionDataset
from utils import AverageMeter
from metric import *
from dataset import get_dataloader
from utils import load_config
import model as module_arch
import model.losses as module_loss
from functools import partial

from accelerate import Accelerator
accelerator = Accelerator()

def parse_arg():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # Param
    parser.add_argument('--dataset-path', default="/public_bme/data/v-lijm/REACT_2024", type=str, help="dataset path")
    parser.add_argument('--split', type=str, help="split of dataset", choices=["val", "test"], required=True)
    parser.add_argument('--resume', default="", type=str, help="checkpoint path")
    parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='mini-batch size (default: 4)')
    parser.add_argument('-j', '--num_workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--img-size', default=256, type=int, help="size of train/test image data")
    parser.add_argument('--crop-size', default=224, type=int, help="crop size of train/test image data")
    parser.add_argument('-max-seq-len', default=751, type=int, help="max length of clip")
    parser.add_argument('--clip-length', default=751, type=int, help="len of video clip")
    parser.add_argument('--window-size', default=8, type=int, help="prediction window-size for online mode")
    parser.add_argument('--feature-dim', default=128, type=int, help="feature dim of model")
    parser.add_argument('--audio-dim', default=78, type=int, help="feature dim of audio")
    parser.add_argument('--_3dmm-dim', default=58, type=int, help="feature dim of 3dmm")
    parser.add_argument('--emotion-dim', default=25, type=int, help="feature dim of emotion")
    parser.add_argument('--online', action='store_true', help='online / offline method')
    parser.add_argument('--outdir', default="./results", type=str, help="result dir")
    parser.add_argument('--device', default='cuda', type=str, help="device: cuda / cpu")
    parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--kl-p', default=0.0002, type=float, help="hyperparameter for kl-loss")
    parser.add_argument('--threads', default=8, type=int, help="num max of threads")
    parser.add_argument('--binarize', action='store_true', help='binarize AUs output from model')

    args = parser.parse_args()
    return args

# Evaluating
def val(args, model, val_loader, criterion, render, binarize=False):
    losses = AverageMeter()

    model, val_loader = accelerator.prepare(model, val_loader)
    model.eval()

    listener_emotion_gt_list = []
    listener_emotion_pred_list = []
    speaker_emotion_list = []
    all_listener_emotion_pred_list = []

    for batch_idx, (s_video, l_video, s_audio, l_audio, s_emotion, s_3dmm, l_emotion, l_3dmm, _, _, _) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            prediction = model(listener_video=l_video, listener_audio=l_audio, 
                           listener_3dmm=l_3dmm, listener_emotion=l_emotion,
                           speaker_video=s_video, speaker_audio=s_audio,
                           speaker_3dmm=s_3dmm, speaker_emotion=s_emotion)

            listener_emotion_out = prediction["pred_emotion"]

            loss = criterion(**prediction)["loss"].item()
            losses.update(loss)
            
            # binarize first 15 positions
            if binarize:
                listener_emotion_out[:, :, :15] = torch.round(listener_emotion_out[:, :, :15])

            listener_emotion_pred_list.append(listener_emotion_out.cpu())
            listener_emotion_gt_list.append(l_emotion.cpu())
            speaker_emotion_list.append(s_emotion.cpu())

    listener_emotion_pred = torch.cat(listener_emotion_pred_list, dim = 0)
    listener_emotion_gt = torch.cat(listener_emotion_gt_list, dim = 0)
    speaker_emotion_gt = torch.cat(speaker_emotion_list, dim = 0)
    all_listener_emotion_pred_list.append(listener_emotion_pred.unsqueeze(1))

    print("-----------------Repeat 9 times-----------------")
    for i in range(9):
        print("Repeat {} times".format(i+1))
        listener_emotion_pred_list = []
        for batch_idx, (s_video, l_video, s_audio, l_audio, s_emotion, s_3dmm, l_emotion, l_3dmm, _, _, _) in enumerate(tqdm(val_loader)):

            with torch.no_grad():
                prediction = model(listener_video=l_video, listener_audio=l_audio, 
                           listener_3dmm=l_3dmm, listener_emotion=l_emotion,
                           speaker_video=s_video, speaker_audio=s_audio,
                           speaker_3dmm=s_3dmm, speaker_emotion=s_emotion)
                listener_emotion_out = prediction["pred_emotion"]

                # binarize first 15 positions
                if binarize:
                    listener_emotion_out[:, :, :15] = torch.round(listener_emotion_out[:, :, :15])
                    
                listener_emotion_pred_list.append(listener_emotion_out.cpu())

        listener_emotion_pred = torch.cat(listener_emotion_pred_list, dim=0)
        all_listener_emotion_pred_list.append(listener_emotion_pred.unsqueeze(1))
        
    all_listener_emotion_pred = torch.cat(all_listener_emotion_pred_list, dim=1)

    print("-----------------Evaluating Metric-----------------")

    p = args.threads
    # If you have problems running function compute_FRC_mp, please replace this function with function compute_FRC
    FRC = compute_FRC_mp(args, all_listener_emotion_pred, listener_emotion_gt, val_test=args.split, p=p)
    print ("FRC: ", FRC)

    # If you have problems running function compute_FRD_mp, please replace this function with function compute_FRD
    FRD = compute_FRD_mp(args, all_listener_emotion_pred, listener_emotion_gt, val_test=args.split, p=p)
    print("FRD: ", FRD)

    FRDvs = compute_FRDvs(all_listener_emotion_pred)
    print("FRDvs: ", FRDvs)
    FRVar  = compute_FRVar(all_listener_emotion_pred)
    print("FRVar: ", FRVar)
    smse  = compute_s_mse(all_listener_emotion_pred)
    print("FRDiv: ", smse)

    # If you have problems running function compute_TLCC_mp, please replace this function with function compute_TLCC
    TLCC = compute_TLCC_mp(all_listener_emotion_pred, speaker_emotion_gt, p=p)

    return losses.avg, FRC, FRD, FRDvs, FRVar, smse, TLCC



def main(args):
    checkpoint_path = args.resume
    config_path = os.path.join(os.path.dirname(checkpoint_path), "config.yaml")
    if not os.path.exists(config_path): # args-based loading --> Trans-VAE by default
        pass
    else: # config-based loading --> BeLFusion
        cfg = load_config(config_path)
        dataset_cfg = cfg.validation_dataset if args.split == "val" else cfg.test_dataset
        dataset_cfg.dataset_path = args.dataset_path
        data_loader = get_dataloader(dataset_cfg, args.split, 
                                    load_audio_s=True, load_audio_l=True, load_video_s=True, load_video_l=True,
                                    load_emotion_s=True, load_emotion_l=True, load_3dmm_s=True, load_3dmm_l=True, load_ref=False,
                                    repeat_mirrored=True)
        model = getattr(module_arch, cfg.arch.type)(cfg.arch.args)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        criterion = partial(getattr(module_loss, cfg.loss.type), **cfg.loss.args)

    if args.resume != '': #  resume from a checkpoint
        print("Resume from {}".format(checkpoint_path))
        checkpoints = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        state_dict = checkpoints['state_dict']
        model.load_state_dict(state_dict)

    render = 0

    val_loss, FRC, FRD, FRDvs, FRVar, smse, TLCC = val(args, model, data_loader, criterion, render, binarize=args.binarize)
    print("{}_loss: {:.5f}".format(args.split, val_loss))
    print("Metric: | FRC: {:.5f} | FRD: {:.5f} | FRDiv: {:.5f} | FRVar: {:.5f} | FRDvs: {:.5f} | FRSyn: {:.5f}".format(FRC, FRD, smse, FRVar, FRDvs, TLCC))
    print("Latex-friendly --> model_name & {:.2f} & {:.2f} & {:.4f} & {:.4f} & {:.4f} & - & {:.2f} \\\\".format( FRC, FRD, smse, FRVar, FRDvs, TLCC))


# ---------------------------------------------------------------------------------


if __name__=="__main__":
    args = parse_arg()
    os.environ["NUMEXPR_MAX_THREADS"] = '16'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    main(args)
