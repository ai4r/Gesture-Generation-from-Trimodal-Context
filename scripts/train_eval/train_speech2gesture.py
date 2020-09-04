import torch
import torch.nn.functional as F


def train_iter_speech2gesture(args, in_spec, target_poses, pose_decoder, discriminator,
                              pose_dec_optim, dis_optim, loss_fn):
    # generation
    pre_poses = target_poses[:, 0:args.n_pre_poses]
    out_poses = pose_decoder(in_spec, pre_poses)

    # to motion
    target_motion = target_poses[:, 1:] - target_poses[:, :-1]
    out_motion = out_poses[:, 1:] - out_poses[:, :-1]

    ###########################################################################################
    # train D
    dis_optim.zero_grad()
    dis_real = discriminator(target_motion)
    dis_fake = discriminator(out_motion.detach())
    dis_error = F.mse_loss(torch.ones_like(dis_real), dis_real) + F.mse_loss(torch.zeros_like(dis_fake), dis_fake)

    dis_error.backward()
    dis_optim.step()

    ###########################################################################################
    # train G
    pose_dec_optim.zero_grad()
    l1_loss = loss_fn(out_poses, target_poses)
    dis_output = discriminator(out_motion)
    gen_error = F.mse_loss(torch.ones_like(dis_output), dis_output)

    loss = args.loss_regression_weight * l1_loss + args.loss_gan_weight * gen_error
    loss.backward()
    pose_dec_optim.step()

    return {'loss': args.loss_regression_weight * l1_loss.item(), 'gen': args.loss_gan_weight * gen_error.item(),
            'dis': dis_error.item()}
