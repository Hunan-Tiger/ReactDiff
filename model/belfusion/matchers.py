import torch
import torch.nn as nn
import os
import numpy as np
from model.belfusion.mlp_diff import MLPSkipNet, Activation
from model.belfusion.unet_diff import GDUnet_Latent
from utils import load_config_from_file
import model as module_arch
from model.belfusion.diffusion import LatentDiffusion
from model.belfusion.resample import UniformSampler

class BaseLatentModel(nn.Module):
    def __init__(self, cfg, emb_size=None, autoencoder_path=None,
                 freeze_encoder=True, cond_embed_dim=None, emb_preprocessing="normalize", **kwargs
                ):
        super(BaseLatentModel, self).__init__()

        self.diffusion = LatentDiffusion(cfg) # TODO init the diffusion object here
        self.schedule_sampler = UniformSampler(self.diffusion)
        
        self.emb_size = emb_size
        self.cond_embed_dim = cond_embed_dim
        self.window_size = cfg.window_size

        def_dtype = torch.get_default_dtype()
        # load auxiliary model
        configpath = os.path.join(os.path.dirname(autoencoder_path), "config.yaml")
        cfg = load_config_from_file(configpath)
        self.embed_ = getattr(module_arch, cfg.arch.type)(cfg.arch.args)

        checkpoint = torch.load(autoencoder_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        self.embed_.load_state_dict(state_dict)
        self.encoder_stats = None #checkpoint["statistics"] if checkpoint["statistics"] is not None else None

        if freeze_encoder:
            for para in self.embed_.parameters():
                para.requires_grad = False
        # if freeze_encoder:
        #     for para in self.embed_.encode.parameters():
        #         para.requires_grad = False
        
        self.emb_preprocessing = emb_preprocessing

        self.linear = nn.Sequential(
            nn.Flatten(-2, -1),
            nn.Linear(emb_size*self.window_size, self.cond_embed_dim),
            nn.GroupNorm(32, self.cond_embed_dim),
            nn.SiLU(inplace=True),
        ) 
    
        torch.set_default_dtype(def_dtype) # config loader changes this


    def preprocess(self, emb): # emb = listener emotion
        stats = self.encoder_stats
        if stats is None:
            return emb # when no checkpoint was loaded, there is no stats.
        if "standardize" in self.emb_preprocessing:
            return (emb - stats["mean"]) / torch.sqrt(stats["var"])
        elif "normalize" in self.emb_preprocessing: # apply
            smooth = 0.0001
            return 2 * (emb - stats["min"]) + smooth / (stats["max"] - stats["min"] + smooth) - 1 
        elif "none" in self.emb_preprocessing.lower():
            return emb
        else:
            raise NotImplementedError(f"Error on the embedding preprocessing value: '{self.emb_preprocessing}'")

    def undo_preprocess(self, emb):
        stats = self.encoder_stats
        if stats is None:
            return emb # when no checkpoint was loaded, there is no stats.

        if "standardize" in self.emb_preprocessing:
            return torch.sqrt(stats["var"]) * emb + stats["mean"]
        elif "normalize" in self.emb_preprocessing:
            return (emb + 1) * (stats["max"] - stats["min"]) / 2 + stats["min"]
        elif "none" in self.emb_preprocessing.lower():
            return emb
        else:
            raise NotImplementedError(f"Error on the embedding preprocessing value: '{self.emb_preprocessing}'")

    def encode(self, video, audio):
        #return self.preprocess(self.embed_.encode(video, audio))
        return self.embed_.encode(video, audio)

    def decode(self, em_emb):
        #return self.embed_.decode(self.undo_preprocess(em_emb))
        return self.embed_.decode(em_emb)
    
    def cond_avg(self, cond):
        '''
        cond: [batch_size, seq_length, features]
        '''
        cond = self.linear(cond)
        return cond

    def get_emb_size(self):
        return self.emb_size

    def forward(self, pred, timesteps, seq_em):
        raise NotImplementedError("This is an abstract class.")
    
    # override checkpointing
    def state_dict(self):
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def to(self, device):
        self.model = self.model.to(device)
        self.embed_ = self.embed_.to(device)
        if self.encoder_stats is not None:
            for key in self.encoder_stats:
                self.encoder_stats[key] = self.encoder_stats[key].to(device)
        super().to(device)
        return self
    
    def cuda(self):
        return self.to(torch.device("cuda"))
    
    # override eval and train
    def train(self, mode=True):
        self.model.train(mode)

    def eval(self):
        self.model.eval()


class LatentMLPMatcher(BaseLatentModel):

    def __init__(self, cfg):
        super(LatentMLPMatcher, self).__init__(cfg, 
                emb_size=cfg.emb_length, autoencoder_path=cfg.autoencoder_path, freeze_encoder=cfg.freeze_encoder, cond_embed_dim=cfg.cond_embed_dim)

        assert cfg.emb_length is not None, "Embedding length must be specified."
        self.emb_length = cfg.emb_length # TODO multiply by 2 if using speaker + listener embeddings

        self.window_size = cfg.window_size
        self.cond_embed_dim = cfg.cond_embed_dim
        self.model_channels = cfg.model_channels
        self.online = cfg.online
        self.seq_len = cfg.seq_len
        self.k = cfg.get("k", 1)
        

        self.init_params = {
            # -------------- 1d unet ---------------
            "in_channels": cfg.get("in_channels", self.seq_len),
            "out_channels": cfg.get("out_channels", self.seq_len),
            "cond_embed_dim": cfg.get("cond_embed_dim", self.cond_embed_dim),
            "model_channels": cfg.get("model_channels", self.model_channels),
            "dims": cfg.get("dims", 1),
            "use_scale_shift_norm": cfg.get("use_scale_shift_norm", False),
            # -------------- mlp ---------------
            "num_channels": self.emb_length,
            "skip_layers": cfg.get("skip_layers", "all"),
            "num_hid_channels": cfg.get("num_hid_channels", 2048),
            "num_layers": cfg.get("num_layers", 20),
            "num_time_emb_channels": cfg.get("num_time_emb_channels", 64),
            "num_cond_emb_channels": self.emb_length, # same as num_channels because same embedder is used for speaker and listener
            "activation": cfg.get("activation", Activation.silu),
            "use_norm": cfg.get("use_norm", True),
            "condition_bias": cfg.get("condition_bias", 1),
            "dropout": cfg.get("dropout", 0),
            "last_act": cfg.get("last_act", Activation.none),
            "num_time_layers": cfg.get("num_time_layers", 2),
            "num_cond_layers": cfg.get("num_emotion_layers", 2),
            "time_last_act": cfg.get("time_last_act", False),
            "cond_last_act": cfg.get("cond_last_act", False),
            "dtype": cfg.get("dtype", "float32")
        }


        self.model = GDUnet_Latent(**self.init_params)
        #self.model = MLPSkipNet(**self.init_params)


    def forward_offline(self, speaker_video=None, speaker_audio=None, speaker_3dmm=None, speaker_emotion=None, listener_video=None, listener_audio=None, listener_3dmm=None, listener_emotion=None, **kwargs):
        is_training = self.model.training
        k_active = self.training

        if is_training:
            batch_size, seq_len = listener_3dmm.shape[:2]
            window_start = torch.randint(0, self.seq_len-self.window_size+1, (1,), device=listener_3dmm.device)
            window_end = window_start + self.window_size
            # target to be predicted (and forward diffused)

            embed_s = self.encode(speaker_video, speaker_audio).reshape(batch_size*(seq_len//self.window_size), self.window_size, -1)
            embed_l = self.encode(listener_video, listener_audio).reshape(batch_size*(seq_len//self.window_size), self.window_size, -1)

            model_cond = self.cond_avg(embed_s) # [B*n, 50, 128] -> [B*n, cond_embed_dim]

            x_start = embed_l
            t, _ = self.schedule_sampler.sample(batch_size*(seq_len//self.window_size), listener_3dmm.device) 

            model_output, loss_ldm = self.diffusion.train_(self.model, x_start, t, model_kwargs={"cond": model_cond}, k_active=k_active)
            model_output = model_output.repeat_interleave(self.k, dim=0) if k_active else model_output

            pred_3dmm, pred_emotion = self.decode(model_output)

            results = {                            
              "pred_emotion": pred_emotion,                     
              "target_emotion": listener_emotion, 
              "pred_3dmm": pred_3dmm,
              "target_3dmm": listener_3dmm,
              "loss_ldm": loss_ldm,
            }

            if k_active and self.k>1:
                results = {k: v.view(batch_size, self.k, *results[k].shape[1:]) for k, v in results.items()}
            return results

        else: # iterate over all windows
            batch_size, seq_len = listener_3dmm.shape[:2]

            embed_s = self.encode(speaker_video, speaker_audio).reshape(batch_size*(seq_len//self.window_size), self.window_size, -1)
            model_cond = self.cond_avg(embed_s)
            
            output = self.diffusion.test_(self, self.model, batch_size*(seq_len//self.window_size), self.window_size, model_kwargs={"cond": model_cond})

            pred_3dmm, pred_emotion = self.decode(output["z_prediction"])
            pred_3dmm = pred_3dmm.reshape(batch_size, seq_len, -1)
            pred_emotion = pred_emotion.reshape(batch_size, seq_len, -1)

            output["pred_3dmm"]=pred_3dmm
            output["target_3dmm"]=listener_3dmm
            output["pred_emotion"]=pred_emotion
            output["target_emotion"]=listener_emotion

            return output


    def forward_online(self, speaker_video=None, speaker_audio=None, speaker_3dmm=None, speaker_emotion=None, listener_video=None, listener_audio=None, listener_3dmm=None, listener_emotion=None, **kwargs):
        is_training = self.model.training
        batch_size = speaker_video.shape[0]

        if is_training:
            # same as offline, but speaker emotion must be shifted by the window size
            # in order to only use past information
            speaker_video_shifted = speaker_video[:, :-self.window_size]
            speaker_audio_shifted = speaker_audio[:, :-self.window_size]

            listener_video_shifted = listener_video[:, self.window_size:]
            listener_audio_shifted = listener_audio[:, self.window_size:]

            listener_3dmm = listener_3dmm[:, self.window_size:]
            listener_emotion = listener_emotion[:, self.window_size:]
            # for the same listener window to be predicted, the speaker emotion will correspond to the past
            return self.forward_offline(speaker_video=speaker_video_shifted, speaker_audio=speaker_audio_shifted,
                                        listener_video=listener_video_shifted, listener_audio=listener_audio_shifted, 
                                        listener_3dmm=listener_3dmm, listener_emotion=listener_emotion, **kwargs)

        else:
            # shift speaker emotion by window size and fill with zeros on the left
            # TODO an alternative strategy might be filling it with the most common speaker emotion
            speaker_video_shifted = torch.cat([torch.zeros_like(speaker_video[:, :self.window_size]), speaker_video[:, :-self.window_size]], dim=1)
            speaker_audio_shifted = torch.cat([torch.zeros_like(speaker_audio[:, :self.window_size]), speaker_audio[:, :-self.window_size]], dim=1)

            return self.forward_offline(speaker_video=speaker_video_shifted, speaker_audio=speaker_audio_shifted, listener_video=listener_video, listener_audio=listener_audio, 
            listener_3dmm=listener_3dmm, listener_emotion=listener_emotion,**kwargs)

    def forward(self, **kwargs):
        if self.online:
            return self.forward_online(**kwargs)
        else:
            return self.forward_offline(**kwargs)