import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================ ReactDiff losses ====================================
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


def MMT_Loss(pred_3dmm, target_3dmm, pred_emotion, target_emotion,
                  w_emo=10, w_coeff=10, w_kld=1,
                  **kwargs):
    # loss for autoencoder. prediction and target have shape of [batch_size, seq_length, features]

    COEFF = eval('MSELoss')(pred_3dmm, target_3dmm, reduction="mean")
    EMO = eval('L1Loss')(pred_emotion, target_emotion, reduction="mean")

    loss_r = w_emo * EMO + w_coeff * COEFF 
    return {"loss": loss_r, "emo": EMO,  "coeff": COEFF}


def CONSINLoss(x, y, dim, reduction="mean", **kwargs):
    loss = 1-torch.cosine_similarity(x, y, dim=dim, eps=1e-08)#.mean(axis=-1)
    if reduction == "mean":
      loss = torch.mean(loss)
    elif reduction == "min":
        loss = loss.min(axis=-1)[0].mean()
    else:
        raise NotImplementedError("reduction {} not implemented".format(reduction))
    
    return loss


def ReactDiffLoss(pred_3dmm, target_3dmm, pred_emotion, target_emotion, split='val', loss_ldm=0., embed_z=None, pred_embed_z=None, losses_multipliers = 1, **kwargs):

    # compute losses
    losses_dict = {"loss": 0}
    if split == 'train':
        losses_dict["epsLoss"] = loss_ldm * losses_multipliers
        losses_dict["loss"] += losses_dict["epsLoss"]

        # ldm只要训练epsloss就好了
        # losses_dict["zLoss"] = eval('CONSINLoss')(pred_embed_z, embed_z, dim=-1, reduction="mean") * losses_multipliers
        # losses_dict["loss"] += losses_dict["zLoss"]

    if split == 'val' or split == 'test':
        losses_dict['emo'] = eval('L1Loss')(pred_emotion, target_emotion, reduction="mean")
        losses_dict['coeff'] = eval('MSELoss')(pred_3dmm, target_3dmm, reduction="mean")

        losses_dict["loss"] += losses_dict['emo'] 
        losses_dict["loss"] += losses_dict['coeff']

    return losses_dict


