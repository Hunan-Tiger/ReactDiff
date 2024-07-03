import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

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


#-------------features_extractors.py----------------
class Attention(nn.Module):
    def __init__(self, embedding_dim=768, dimk=768, dimv=768, hidden_dim=512, dropout=0.1):
        super(Attention, self).__init__()
        self.q = nn.Linear(embedding_dim, hidden_dim) if hidden_dim != embedding_dim else nn.Identity()
        self.k = nn.Linear(dimk, hidden_dim) if hidden_dim != dimk else nn.Identity()
        self.v = nn.Linear(dimv, hidden_dim) if hidden_dim != dimv else nn.Identity()

        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, query, key, value):
        q = self.q(query)
        k = self.k(key)
        v = self.v(value)
        attention_output, _ = self.attention(q, k, v)
        attention_output = self.norm1(attention_output + q)
        # attention_output = self.norm2(self.drop(attention_output + self.feedforward(attention_output)))
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

        self.fusion_0 = nn.Linear(64, self.hidden_dim)
        self.fusion_1 = nn.Linear(25088, self.hidden_dim)
        self.fusion_2 = nn.Linear(1408, self.hidden_dim)

        self.fusion_all = nn.Linear(self.hidden_dim*3, self.hidden_dim)

    def forward(self, video):
        video_va = self.fusion_0(video[:,:,:64])
        video_au = self.fusion_1(video[:,:,64:64+25088])
        video_fer = self.fusion_2(video[:,:,64+25088:])
        video_input = torch.cat([video_va, video_au, video_fer], dim=-1)
        Video_features_fusion = self.fusion_all(video_input)

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

        self.audio_feature_fusion = nn.Linear(self.audio_dim+26, self.hidden_dim)

        self.video_audio_fusion = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.video_audio_fusion_map_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim*2),
            nn.LayerNorm(self.hidden_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim*2, self.feature_dim)
        )


    def forward(self, video, audio):
        batch_size, frame_num = video.shape[:2]
        video_feature = self.video_features_fusion(video) # B, seq_len, hidden_dim

        audio_feature = self.audio_feature_fusion(audio)  # B, seq_len, 1024

        feature_ = self.video_audio_fusion(torch.cat([video_feature, audio_feature], dim=-1))

        speaker_behaviour_feature = self.video_audio_fusion_map_layer(feature_).reshape(batch_size * (frame_num // self.window_size), self.window_size, -1) 

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
        self.linear = nn.Linear(in_channels, latent_dim)
        self.PE = PositionalEncoding(latent_dim)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                             nhead=4,
                                                             dim_feedforward=latent_dim * 2,
                                                             dropout=0.1,batch_first=True)

        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer, num_layers=1)
        self.mu_token = nn.Parameter(torch.randn(latent_dim))  ## [0,1)标准正态分布随机数
        self.logvar_token = nn.Parameter(torch.randn(latent_dim))

        self.fc_x_enc = nn.Linear(self.window_size, 1)
        self.fc_mu_enc = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_logvar_enc = nn.Linear(self.latent_dim, self.latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input):
        x = self.linear(input)  # B, seq_len（256）, latent_dim（128）
        B, T, D = input.shape
        # x = self.PE(x)

        # lengths = [len(item) for item in input]  ## [T, T, ..] B个T

        # token_mask = torch.ones((B, 2), dtype=bool, device=input.device)  ## (B, 2),全为True
        # mask = lengths_to_mask(lengths, input.device)   ## (B, seq_len),全为True
        # aug_mask = torch.cat((token_mask, mask), 1) ## augmentation  (B, 2+seq_len),True

        # x = self.seqTransEncoder(x, src_key_padding_mask=~mask).permute(0, 2, 1)

        x = self.fc_x_enc(x.permute(0, 2, 1)).squeeze(-1)  # [bs*n, 128]

        mu = self.fc_mu_enc(x)  # mu_token [B, 128]
        logvar = self.fc_logvar_enc(x)  # log_var [B, 128]
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        return z, mu, logvar



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

        decoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim,
                                                             nhead=n_head,
                                                             dim_feedforward=feature_dim * 2,
                                                             dropout=0.1, batch_first=True)
        self.listener_reaction_decoder = nn.TransformerEncoder(decoder_layer, num_layers=2)


        self.biased_mask = init_biased_mask(n_head = n_head, max_seq_len = seq_len, period=seq_len)

        self.listener_reaction_3dmm_map_layer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim*2),
            nn.LayerNorm(feature_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim*2, output_3dmm_dim)
        )
        self.listener_reaction_emotion_map_layer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim*2),
            nn.LayerNorm(feature_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim*2, output_emotion_dim)
        )
        self.PE = PositionalEncoding(feature_dim)


    def forward(self, z):
        
        z = z.unsqueeze(1).repeat(1, self.window_size, 1)if len(z.shape)==2 else z# [bs*n, fdim] -> [bs*n, ws, fdim]

        listener_reaction = z

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
        self.coeff_3dmm = cfg.coeff_3dmm
        self.emotion = cfg.emotion

        self.seq_len = cfg.seq_len
        self.depth = cfg.depth
        self.device = cfg.device

        self.Behaviour_features_fusion = Behaviour_features_fusion_block(video_dim=self.video_dim, audio_dim=self.audio_dim, hidden_dim=self.hidden_dim, window_size=self.window_size, feature_dim=self.feature_dim, depth=self.depth, device=self.device)

        self.reaction_encoder = Encoder(in_channels=self.feature_dim, latent_dim=self.feature_dim, device=self.device)

        self.reaction_decoder = Decoder(output_3dmm_dim = self.output_3dmm_dim, output_emotion_dim = self.output_emotion_dim, feature_dim = self.feature_dim, seq_len=self.seq_len, device=self.device)


    # ---------self training----------------
    def _encode(self, a, b):
        batch_size, frame_num = a.shape[:2]
        z, mu, logvar = self.reaction_encoder(self.Behaviour_features_fusion(a, b).reshape(batch_size*(frame_num // self.window_size), self.window_size, -1))
        return z, mu, logvar
    
    def _decode(self, z):
        pred_3dmm, pred_emotion = self.reaction_decoder(z)
        return pred_3dmm, pred_emotion
    
    
    # -------------------- ldm -------------------
    def encode(self, a, b):
        z, _, _ = self._encode(a, b)
        return z
    
    def decode(self, z):
        y_3dmm, y_emotion = self._decode(z)
        return y_3dmm, y_emotion
    

    def forward(self, listener_video=None, listener_audio=None, listener_3dmm=None, listener_emotion=None, **kwargs):

        features_fusion, mu, logvar = self._encode(listener_video, listener_audio)

        seq_len = listener_video.shape[1]
        pred_3dmm, pred_emotion = self._decode(features_fusion)
        pred_3dmm = pred_3dmm.reshape(-1, seq_len, self.output_3dmm_dim)
        pred_emotion = pred_emotion.reshape(-1, seq_len, self.output_emotion_dim)
        target_3dmm, target_emotion = listener_3dmm, listener_emotion

        result = {'pred_emotion':pred_emotion, 'target_emotion':target_emotion, 'pred_3dmm':pred_3dmm, 'target_3dmm':target_3dmm, 'mu':mu, 'logvar':logvar}
        return result



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    l_video = torch.randn(2, 750, 64+25088+1408)
    l_audio = torch.randn(2, 750, 1536)
    l_emotion = torch.randn(2, 750, 25)
    l_3dmm = torch.randn(2, 750, 58)
    model = TransformerVAE()
    res = model(l_video, l_audio, l_emotion, l_3dmm)
    print(res['prediction'].shape, res['target'].shape, res['coefficients_3dmm'].shape, res['target_coefficients'].shape)
