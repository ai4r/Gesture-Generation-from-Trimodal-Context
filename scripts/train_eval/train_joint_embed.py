import torch
import torch.nn.functional as F


def train_iter_embed(args, epoch, in_text, in_audio, target_data, net, optim, mode=None):
    pre_seq = target_data[:, 0:args.n_pre_poses]

    # zero gradients
    optim.zero_grad()

    if mode == 'random':  # joint embed model
        variational_encoding = False  # AE
    else:  # feature extractor in FGD
        variational_encoding = False  # VAE or AE

    # reconstruction loss
    context_feat, context_mu, context_logvar, poses_feat, pose_mu, pose_logvar, recon_data = \
        net(in_text, in_audio, pre_seq, target_data, mode, variational_encoding=variational_encoding)

    recon_loss = F.l1_loss(recon_data, target_data, reduction='none')
    recon_loss = torch.mean(recon_loss, dim=(1, 2))

    if False:  # use pose diff
        target_diff = target_data[:, 1:] - target_data[:, :-1]
        recon_diff = recon_data[:, 1:] - recon_data[:, :-1]
        recon_loss += torch.mean(F.l1_loss(recon_diff, target_diff, reduction='none'), dim=(1, 2))

    recon_loss = torch.sum(recon_loss)

    # KLD
    if variational_encoding:
        if net.mode == 'speech':
            KLD = -0.5 * torch.sum(1 + context_logvar - context_mu.pow(2) - context_logvar.exp())
        else:
            KLD = -0.5 * torch.sum(1 + pose_logvar - pose_mu.pow(2) - pose_logvar.exp())

        if epoch < 10:
            KLD_weight = 0
        else:
            KLD_weight = min(1.0, (epoch - 10) * args.loss_kld_weight)
        loss = args.loss_regression_weight * recon_loss + KLD_weight * KLD
    else:
        loss = recon_loss

    loss.backward()
    optim.step()

    ret_dict = {'loss': recon_loss.item()}
    if variational_encoding:
        ret_dict['KLD'] = KLD.item()
    return ret_dict


def eval_embed(in_text, in_audio, pre_poses, target_poses, net, mode=None):
    context_feat, context_mu, context_logvar, poses_feat, pose_mu, pose_logvar, recon_poses = \
        net(in_text, in_audio, pre_poses, target_poses, mode, variational_encoding=False)

    recon_loss = F.l1_loss(recon_poses, target_poses, reduction='none')
    recon_loss = torch.mean(recon_loss, dim=(1, 2))
    loss = torch.mean(recon_loss)

    return loss, recon_poses
