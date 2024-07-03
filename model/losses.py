import torch
import torch.nn as nn
import torch.nn.functional as F



class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, q, p):
        div = torch.distributions.kl_divergence(q, p)# KL散度可以用来衡量两个概率分布之间的相似性，两个概率分布越相近，KL散度越小。
        return div.mean()

    def __repr__(self):
        return "KLLoss()"



class VAELoss(nn.Module):
    def __init__(self, kl_p=0.0002):
        super(VAELoss, self).__init__()
        self.mse = nn.MSELoss(reduce=True, size_average=True)
        self.kl_loss = KLLoss()
        self.kl_p = kl_p

    def forward(self, gt_emotion, gt_3dmm, pred_emotion, pred_3dmm, distribution):
        rec_loss = self.mse(pred_emotion, gt_emotion) + self.mse(pred_3dmm[:,:, :52], gt_3dmm[:,:, :52]) + 10*self.mse(pred_3dmm[:,:, 52:], gt_3dmm[:,:, 52:])  ## 元素之间的平均平方误差
        mu_ref = torch.zeros_like(distribution[0].loc).to(gt_emotion.get_device())
        scale_ref = torch.ones_like(distribution[0].scale).to(gt_emotion.get_device())
        distribution_ref = torch.distributions.Normal(mu_ref, scale_ref) # 标准正态分布, [128]

        kld_loss = 0
        for t in range(len(distribution)):
            kld_loss += self.kl_loss(distribution[t], distribution_ref) # 让128维特征编码的分布接近标准正态分布
        kld_loss = kld_loss / len(distribution)

        loss = rec_loss + self.kl_p * kld_loss


        return loss, rec_loss, kld_loss

    def __repr__(self):
        return "VAELoss()"



def div_loss(Y_1, Y_2):
    loss = 0.0
    b,t,c = Y_1.shape
    Y_g = torch.cat([Y_1.view(b,1,-1), Y_2.view(b,1,-1)], dim = 1)
    for Y in Y_g:
        dist = F.pdist(Y, 2) ** 2
        loss += (-dist /  100).exp().mean()
    loss /= b
    return loss




# ================================ BeLFUSION losses ====================================

def TemporalLoss(prediction, target, **kwargs):
    # prediction has shape of [batch_size, seq_length, features]
    # target has shape of [batch_size, seq_length, features]
    batch_size, seq_len, _= prediction.shape

    # 相邻帧差值，即输出视频要满足GT的时序变换
    prediction = prediction.reshape(batch_size*seq_len, -1)
    target = target.reshape(batch_size*seq_len, -1)
    tem_loss = 0
    for t in range(1, prediction.shape[0]):
        if t % seq_len == 0:
            continue
        tem_loss += abs((prediction[t] - prediction[t-1]) - (target[t] - target[t-1])).mean()
    tem_loss = tem_loss / batch_size

    return tem_loss


def Transvae_Loss(pred_3dmm, target_3dmm, pred_emotion, target_emotion,
                  w_emo=10, w_coeff=10, w_kld=1,
                  **kwargs):
    # loss for autoencoder. prediction and target have shape of [batch_size, seq_length, features]

    COEFF = eval('MSELoss')(pred_3dmm, target_3dmm, reduction="mean")
    EMO = eval('L1Loss')(pred_emotion, target_emotion, reduction="mean")

    loss_r = w_emo * EMO + w_coeff * COEFF 
    return {"loss": loss_r, "emo": EMO,  "coeff": COEFF}



# ---------------------FaRfusion Loss---------------------------
def MSELoss(prediction, target, reduction="mean", **kwargs):
    # prediction has shape of [batch_size, num_preds, features]
    # target has shape of [batch_size, num_preds, features]
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
    assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, num_preds, features]"

    # manual implementation of MSE loss
    loss = ((prediction - target) ** 2).mean(axis=-1) ## 按列返回mean [batch_size, num_preds, features] -> [batch_size, num_preds]
    
    # reduce across multiple predictions
    if reduction == "mean":
        loss = torch.mean(loss) ## 返回所有元素的mean
    elif reduction == "min":
        loss = loss.min(axis=-1)[0].mean()
    else:
        raise NotImplementedError("reduction {} not implemented".format(reduction))
    return loss

def L1Loss(prediction, target, reduction="mean", **kwargs):
    # prediction has shape of [batch_size, num_preds, features]
    # target has shape of [batch_size, num_preds, features]
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
    assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, num_preds, features]"

    # manual implementation of L1 loss
    loss = (torch.abs(prediction - target)).mean(axis=-1)
    
    # reduce across multiple predictions
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "min":
        loss = loss.min(axis=-1)[0].mean()
    else:
        raise NotImplementedError("reduction {} not implemented".format(reduction))
    return loss

def CONSINLoss(x, y, dim, reduction="mean", **kwargs):
    loss = 1-torch.cosine_similarity(x, y, dim=dim, eps=1e-08)#.mean(axis=-1)
    if reduction == "mean":
      loss = torch.mean(loss)
    elif reduction == "min":
        loss = loss.min(axis=-1)[0].mean()
    else:
        raise NotImplementedError("reduction {} not implemented".format(reduction))
    
    return loss


def BeLFusionLoss(pred_3dmm, target_3dmm, pred_emotion, target_emotion, split='val', loss_ldm=0., embed_z=None, pred_embed_z=None, losses_multipliers = 1, 
                  **kwargs):

    # compute losses
    losses_dict = {"loss": 0}
    if split == 'train':
        losses_dict["epsLoss"] = loss_ldm * losses_multipliers
        losses_dict["loss"] += losses_dict["epsLoss"]

        # ldm只要训练epsloss就好了
        # losses_dict["zLoss"] = eval('CONSINLoss')(pred_embed_z, embed_z, dim=-1, reduction="mean") * losses_multipliers
        # losses_dict["loss"] += losses_dict["zLoss"]

    if split == 'val':
        losses_dict['MSELoss'] = eval('MSELoss')(pred_3dmm, target_3dmm, reduction="mean")* losses_multipliers
        losses_dict['L1Loss'] = eval('MSELoss')(pred_emotion, target_emotion, reduction="mean")* losses_multipliers

        losses_dict["loss"] += losses_dict['MSELoss'] 
        losses_dict["loss"] += losses_dict['L1Loss']

    return losses_dict