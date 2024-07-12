import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import argparse

# --------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=True):
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, :x.shape[1], :]
        else:
            x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)



def lengths_to_mask(lengths, device):
    lengths = torch.tensor(lengths, device=device)
    max_len = max(lengths) # 255
    mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask



# Temporal Bias, inspired by ALiBi: https://github.com/ofirpress/attention_with_linear_biases
def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask


class Attention(nn.Module):
    def __init__(self, embedding_dim=768, dimk=768, dimv=768, hidden_dim=512, dropout=0.1):
        super(Attention, self).__init__()
        self.q = nn.Linear(embedding_dim, hidden_dim) if hidden_dim != embedding_dim else nn.Identity()
        self.k = nn.Linear(dimk, hidden_dim) if hidden_dim != dimk else nn.Identity()
        self.v = nn.Linear(dimv, hidden_dim) if hidden_dim != dimv else nn.Identity()

        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, query, key, value):
        q = self.q(query)
        k = self.k(key)
        v = self.v(value)
        attention_output, _ = self.attention(q, k, v)
        attention_output = self.norm1(attention_output + q)
        return attention_output


class Block(nn.Module):
    def __init__(self, hidden_dim=512, dimq=25088, dim1=1408, dim2=64):
        super(Block, self).__init__()
        self.dimq = dimq
        self.dim1 = dim1
        self.dim2 = dim2
        self.hidden_dim = hidden_dim
        self.attn1 = Attention(embedding_dim=self.dimq, dimk=self.dim1, dimv=self.dim1, hidden_dim=self.hidden_dim)
        self.attn2 = Attention(embedding_dim=self.hidden_dim, dimk=self.dim2, dimv=self.dim2, hidden_dim=self.hidden_dim)

    def forward(self, query, key_0, value_0, key_1, value_1):
        query = self.attn1(query, key_0, value_0)
        return self.attn2(query, key_1, value_1)

class Video_features_fusion(nn.Module):
    def __init__(self, video_dim = 128, hidden_dim = 1024, depth = 2,device = 'cpu'):
        super(Video_features_fusion, self).__init__()

        self.video_dim = video_dim 
        self.device = device
        self.hidden_dim = hidden_dim

        self.fusion_0 = Attention(embedding_dim=self.video_dim, dimk=self.video_dim, dimv=self.video_dim, hidden_dim=self.hidden_dim)

        self.fusion_cyc = nn.ModuleList([Block(hidden_dim=self.hidden_dim, dimq=self.hidden_dim, dim1=1408, dim2=64) for _ in range(depth)])

    def forward(self, video):
        video_input = self.fusion_0(video[:,:,64:64+25088], video[:,:,64:64+25088], video[:,:,64:64+25088])
        for blk in self.fusion_cyc:
            video_input = blk(video_input, video[:,:,64+25088:], video[:,:,64+25088:], video[:,:,:64], video[:,:,:64])
        Video_features_fusion = video_input

        return  Video_features_fusion


class Behaviour_features_fusion_block(nn.Module):
    def __init__(self, video_dim = 25088, audio_dim = 1536, feature_dim=128,  hidden_dim = 1024, depth = 2, window_size=50, device = 'cpu'):
        super(Behaviour_features_fusion_block, self).__init__()

        self.device = device
        self.hidden_dim = hidden_dim
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.feature_dim = feature_dim
        self.window_size = window_size

        self.video_features_fusion = Video_features_fusion(video_dim=self.video_dim, hidden_dim=self.hidden_dim, depth=depth, device=device)

        self.audio_feature_fusion = Attention(embedding_dim=self.audio_dim, dimk=26, dimv=26, hidden_dim=self.hidden_dim)

        self.video_audio_fusion = nn.ModuleList([Attention(embedding_dim=self.hidden_dim, dimk=self.hidden_dim, dimv=self.hidden_dim, hidden_dim=self.hidden_dim) for _ in range(depth)])
        self.video_audio_fusion_map_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim*2),
            nn.LayerNorm(self.hidden_dim*2),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim*2, self.feature_dim)
        )


    def forward(self, video, audio):
        batch_size, frame_num = video.shape[:2]
        video_feature = self.video_features_fusion(video) # B, seq_len, hidden_dim

        audio_feature = self.audio_feature_fusion(audio[:,:,26:], audio[:,:,:26], audio[:,:,:26])  # B, seq_len, 1024

        for i in self.video_audio_fusion:
            video_feature = i(video_feature, audio_feature, audio_feature)

        speaker_behaviour_feature = self.video_audio_fusion_map_layer(video_feature)

        return  speaker_behaviour_feature  ## B, seq_len, 128 -> B*n, window_size, 128




class Encoder(nn.Module): ## Transformer Encoder
    def __init__(self,
                 in_channels: int,
                 latent_dim: int = 128,
                 window_size: int = 50,
                 **kwargs) -> None:
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.window_size = window_size
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, 2*latent_dim),
            nn.LayerNorm(2*latent_dim),
            nn.SiLU(inplace=True),
            nn.Linear(2*latent_dim, latent_dim),
        )

    def forward(self, input):
        z = self.MLP(input)  # B*n, ws, latent_dim（128）
        return z 



class Decoder(nn.Module):  ## Transformer
    def __init__(self, output_3dmm_dim = 58, output_emotion_dim = 25, feature_dim = 128, device = 'cpu', seq_len=750, n_head = 4, window_size=50, **kwargs):
        super(Decoder, self).__init__()

        self.feature_dim = feature_dim
        self.device = device
        self.seq_len = seq_len
        self.window_size = window_size
        self.output_3dmm_dim = output_3dmm_dim
        self.output_emotion_dim = output_emotion_dim

        self.encoder = Encoder(feature_dim, feature_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim,
                                                             nhead=n_head,
                                                             dim_feedforward=feature_dim * 2,
                                                             dropout=0.1, batch_first=True)
        self.listener_reaction_decoder_1 = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.listener_reaction_decoder_2 = nn.TransformerDecoder(decoder_layer, num_layers=2)


        self.biased_mask = init_biased_mask(n_head = n_head, max_seq_len = seq_len, period=seq_len)

        self.listener_reaction_3dmm_map_layer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim*2),
            nn.LayerNorm(feature_dim*2),
            nn.SiLU(inplace=True),
            nn.Linear(feature_dim*2, output_3dmm_dim)
        )
        self.listener_reaction_emotion_map_layer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim*2),
            nn.LayerNorm(feature_dim*2),
            nn.SiLU(inplace=True),
            nn.Linear(feature_dim*2, output_emotion_dim)
        )

    def forward(self, l=None):

        listener_reaction = self.listener_reaction_decoder_1(l, l)
        listener_reaction = self.listener_reaction_decoder_2(listener_reaction, listener_reaction)

        listener_3dmm_out = self.listener_reaction_3dmm_map_layer(listener_reaction)## [bs*n, ws, 58]
        listener_emotion_out = self.listener_reaction_emotion_map_layer(listener_reaction)

        return listener_3dmm_out, listener_emotion_out



class TransformerVAE(nn.Module):
    def __init__(self, cfg):
        super(TransformerVAE, self).__init__()

        self.feature_dim = cfg.feature_dim
        self.hidden_dim = cfg.hidden_dim
        self.window_size = cfg.window_size
        self.video_dim = cfg.video_dim
        self.audio_dim = cfg.audio_dim


        self.output_3dmm_dim = cfg.output_3dmm_dim
        self.output_emotion_dim = cfg.output_emotion_dim

        self.seq_len = cfg.seq_len
        self.depth = cfg.depth
        self.device = cfg.device

        self.Behaviour_features_fusion = Behaviour_features_fusion_block(video_dim=self.video_dim, audio_dim=self.audio_dim, hidden_dim=self.hidden_dim, window_size=self.window_size, feature_dim=self.feature_dim, depth=self.depth, device=self.device)

        self.reaction_encoder = Encoder(in_channels=self.feature_dim, latent_dim=self.feature_dim, device=self.device)

        self.reaction_decoder = Decoder(output_3dmm_dim = self.output_3dmm_dim, output_emotion_dim = self.output_emotion_dim, feature_dim = self.feature_dim, seq_len=self.seq_len, device=self.device)


    # ---------self training----------------
    def _encode(self, a, b):
        z = self.reaction_encoder(self.Behaviour_features_fusion(a, b))
        return z
    
    def _decode(self, l):
        pred_3dmm, pred_emotion = self.reaction_decoder(l)
        return pred_3dmm, pred_emotion
    
    
    # -------------------- ldm -------------------
    def encode(self, a, b):
        z = self._encode(a, b) #[B, 750, 128]
        return z
    
    def decode(self, z):
        y_3dmm, y_emotion = self._decode(z)
        return y_3dmm, y_emotion
    

    def forward(self, listener_video=None, listener_audio=None, listener_3dmm=None, listener_emotion=None, speaker_video=None, speaker_audio=None,  **kwargs):
        batch_size, seq_len = listener_video.shape[:2]
        l_features_fusion = self._encode(listener_video, listener_audio)

        pred_3dmm, pred_emotion = self._decode(l_features_fusion)

        target_3dmm, target_emotion = listener_3dmm, listener_emotion

        result = {'pred_emotion':pred_emotion, 'target_emotion':target_emotion, 'pred_3dmm':pred_3dmm, 'target_3dmm':target_3dmm}
        return result



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    l_video = torch.randn(2, 750, 64+25088+1408)
    l_audio = torch.randn(2, 750, 1536+26)
    l_emotion = torch.randn(2, 750, 25)
    l_3dmm = torch.randn(2, 750, 58)
    s_video = torch.randn(2, 750, 64+25088+1408)
    s_audio = torch.randn(2, 750, 1536+26)
    def parse_arg():
      parser = argparse.ArgumentParser(description='PyTorch Training')
      # Param
      parser.add_argument('--feature_dim', default=128)
      parser.add_argument('--hidden_dim', default=1024)
      parser.add_argument('--window_size', default=50)
      parser.add_argument('--video_dim', default=25088)
      parser.add_argument('--audio_dim', default=1536)
      parser.add_argument('--output_3dmm_dim', default=58)
      parser.add_argument('--output_emotion_dim', default=25)
      parser.add_argument('--seq_len', default=750)
      parser.add_argument('--depth', default=2)
      parser.add_argument('--device', default='cuda')

      args = parser.parse_args()
      return args
    cfg = parse_arg()
    model = TransformerVAE(cfg)
    res = model(l_video, l_audio, l_3dmm, l_emotion, s_video, l_audio)
    print(res['pred_emotion'].shape, res['target_emotion'].shape, res['pred_3dmm'].shape, res['target_3dmm'].shape)
