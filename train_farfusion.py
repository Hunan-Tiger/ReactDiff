# python train_farfusion.py config=config/1_belfusion_vae.yaml name=TransformerVAE arch.args.online=False
# python train_farfusion.py config=config/2_belfusion_ldm.yaml name=LatentMLPMatcher arch.args.online=False
# python evaluate.py  --resume ./results/LatentMLPMatcher/checkpoint_best.pth  --gpu-ids 0  --split val
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
import logging
import model as module_arch
from metric import *
import model.losses as module_loss
from functools import partial
from utils import load_config, store_config, AverageMeter
from dataset import get_dataloader
import wandb
from datetime import datetime
import random

from accelerate import Accelerator
accelerator = Accelerator()


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def pprint(f, msg):
    print(msg)
    f.write(msg + "\n")


def evaluate(cfg, pred_list_em, speaker_em, listener_em, epoch):
    assert listener_em.shape[0] == speaker_em.shape[0], "speaker and listener emotion must have the same shape"
    assert listener_em.shape[0] == pred_list_em.shape[0], "predictions and listener emotion must have the same shape"

    # only the fast diversity metrics ploted often
    metrics = {
        # APPROPRIATENESS METRICS
        #"FRDist": compute_FRD(data_path, pred_list_em[:,0], listener_em), # FRDist (1) --> slow, ~3 mins
        #"FRCorr": compute_FRC(data_path, pred_list_em[:,0], listener_em), # FRCorr (2) --> slow, ~3 mins

        # DIVERSITY METRICS --> all very fast, compatible with validation in training loop
        "FRVar": compute_FRVar(pred_list_em), # FRVar (1) --> intra-variance (among all frames in a prediction),
        "FRDiv": compute_s_mse(pred_list_em), # FRDiv (2) --> inter-variance (among all predictions for the same speaker),
        "FRDvs": compute_FRDvs(pred_list_em), # FRDvs (3) --> diversity among reactions generated from different speaker behaviours
        
        # OTHER METRICS
        # FRRea (realism)
        #"FRSyn": compute_TLCC(pred_list_em, speaker_em), # FRSyn (synchrony) --> EXTREMELY slow, ~1.5h
    }
    return metrics


def update_averagemeter_from_dict(results, meters):
    # if meters is empty, it will be initialized. If not, it will be updated
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            value = value.item()

        if key in meters:
            meters[key].update(value)
        else:
            meters[key] = AverageMeter()
            meters[key].update(value)

# Train
def train(cfg, model, train_loader, optimizer, criterion, device, ema):
    losses_meters = {}

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    model.train()
    for batch_idx, (s_video, l_video, s_audio, l_audio, s_emotion, s_3dmm, l_emotion, l_3dmm, l_reference, _, _) in enumerate(tqdm(train_loader)):

        optimizer.zero_grad()

        prediction = model(listener_video=l_video, listener_audio=l_audio, 
                           listener_3dmm=l_3dmm, listener_emotion=l_emotion,
                           speaker_video=s_video, speaker_audio=s_audio,
                           speaker_3dmm=s_3dmm, speaker_emotion=s_emotion)
        prediction["split"] = 'train'

        losses = criterion(**prediction)
        update_averagemeter_from_dict(losses, losses_meters)
        accelerator.backward(losses["loss"])
        optimizer.step()
        ema.update()

    return {key: losses_meters[key].avg for key in losses_meters}


def validate(cfg, model, val_loader, criterion, device, epoch, ema):
    num_preds = cfg.trainer.get("num_preds", 10) # number of predictions to make
    losses_meters = {}

    model, val_loader = accelerator.prepare(model, val_loader)
    ema.apply_shadow()
    model.eval()

    with torch.no_grad():
        for batch_idx, (s_video, l_video, s_audio, l_audio, s_emotion, s_3dmm, l_emotion, l_3dmm, l_reference, _, _) in enumerate(tqdm(val_loader)):
            prediction = model(listener_video=l_video, listener_audio=l_audio, 
                           listener_3dmm=l_3dmm, listener_emotion=l_emotion,
                           speaker_video=s_video, speaker_audio=s_audio,
                           speaker_3dmm=s_3dmm, speaker_emotion=s_emotion) # [B, S, D]
            prediction["split"] = 'val'

            losses = criterion(**prediction)
            update_averagemeter_from_dict(losses, losses_meters)


    return {"val_" + key: losses_meters[key].avg for key in losses_meters}



def compute_statistics(config, model, data_loader, device):
    checkpoint_path = config.resume
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # reload checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    model, data_loader = accelerator.prepare(model, data_loader)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch_idx, (s_video, l_video, s_audio, l_audio, _, _, _, _, _, _, _) in enumerate(tqdm(data_loader)):
            prediction = model.encode(s_video, s_audio)
            preds.append(prediction)
            prediction = model.encode(l_video, l_audio)
            preds.append(prediction)

    preds = torch.cat(preds, axis=0)

    checkpoint["statistics"] = {
        "min": preds.min(axis=0).values,
        "max": preds.max(axis=0).values,
        "mean": preds.mean(axis=0),
        "std": preds.std(axis=0),
        "var": preds.var(axis=0),
    }
    
    torch.save(checkpoint, config.resume)


def main():
    # load yaml config
    cfg = load_config()
    cfg.trainer.out_dir = os.path.join(cfg.trainer.out_dir, cfg["name"])
    os.makedirs(cfg.trainer.out_dir, exist_ok=True)
    store_config(cfg)
    f = open(os.path.join(cfg.trainer.out_dir, "log.txt"), "w")

    start_epoch = 0
    pprint(f, str(cfg.dataset))
    pprint(f, str(cfg.validation_dataset))
    
    train_loader = get_dataloader(cfg.dataset, cfg.dataset.split, 
                                  load_audio_s=True, load_audio_l=True, load_video_s=True, load_video_l=True,
                                  load_emotion_s=True, load_emotion_l=True, load_3dmm_s=True, load_3dmm_l=True, load_ref=False, repeat_mirrored=True)

    valid_loader = get_dataloader(cfg.validation_dataset, cfg.validation_dataset.split, 
                                  load_audio_s=True, load_audio_l=True, load_video_s=True, load_video_l=True,
                                  load_emotion_s=True, load_emotion_l=True, load_3dmm_s=True, load_3dmm_l=True, load_ref=False, repeat_mirrored=True)

    pprint(f, 'Train dataset: {} samples'.format(len(train_loader.dataset)))
    pprint(f, 'Valid dataset: {} samples'.format(len(valid_loader.dataset)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = getattr(module_arch, cfg.arch.type)(cfg.arch.args)

    ema = EMA(model.to(device), 0.999)
    ema.register()

    # model = model.to(device)
    pprint(f, 'Model {} : params: {:4f}M'.format(cfg.arch.type, sum(p.numel() for p in model.parameters()) / 1000000.0))
    
    criterion = partial(getattr(module_loss, cfg.loss.type), **cfg.loss.args)
    optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.999), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)

    if cfg.trainer.resume != None:
        checkpoint_path = cfg.trainer.resume
        pprint(f, "Resume from {}".format(checkpoint_path))
        checkpoints = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoints['state_dict']
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoints['optimizer'])

    last_epoch_stored = 99
    val_loss = 0
    val_metrics = None
    log_dict = {}
    val_loss_max = 999999

    for epoch in range(start_epoch, cfg.trainer.epochs):

        # =================== TRAIN ===================
        train_losses = train(cfg, model, train_loader, optimizer, criterion, device, ema)
        log_dict.update(train_losses)

        # =================== VALIDATION ===================
        if (cfg.trainer.val_period > 0 and (epoch + 1) % cfg.trainer.val_period == 0) or epoch == start_epoch or epoch == last_epoch_stored:
            val_losses = validate(cfg, model, valid_loader, criterion, device, epoch, ema)
            log_dict.update(val_losses)

            print(f"-----------------------Updated best model at epoch_{epoch+1} !-----------------------")
            checkpoint = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    }
            os.makedirs(cfg.trainer.out_dir, exist_ok=True)
            torch.save(checkpoint, os.path.join(cfg.trainer.out_dir, 'checkpoint_epoch{}.pth'.format(epoch+1)))
            ema.restore()

        # =================== log ===================
        log_message = 'epoch: {}'.format(epoch)
        for key, value in log_dict.items():
            log_message += ", {}: {:.6f}".format(key, value)
        pprint(f, log_message)
        f.flush()
    
    # if cfg.arch.type == 'TransformerVAE':
    #     pprint(f, f"Starting stats computation...")
    #     #cfg.resume = os.path.join(cfg.trainer.out_dir, f"checkpoint_{last_epoch_stored}.pth")
    #     cfg.resume = os.path.join(cfg.trainer.out_dir, f"checkpoint_epoch100.pth")
    #     compute_statistics(cfg, model, train_loader, device)
    #     pprint(f, "Stats computed!")
    #     pprint(f, '=' * 80)

    f.close()

# ---------------------------------------------------------------------------------


if __name__=="__main__":
    torch.manual_seed(6)
    torch.cuda.manual_seed(6)
    np.random.seed(6)
    random.seed(6)
    main()