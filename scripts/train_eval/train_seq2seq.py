import logging
import torch
import torch.nn.functional as F

loss_i = 0
def custom_loss(output, target, args, epoch):
    n_element = output.numel()

    # mae
    mse_loss = F.mse_loss(output, target)
    mse_loss *= args.loss_regression_weight

    # continuous motion
    diff = [abs(output[:, n, :] - output[:, n-1, :]) for n in range(1, output.shape[1])]
    cont_loss = torch.sum(torch.stack(diff)) / n_element
    cont_loss *= args.loss_kld_weight

    # motion variance
    norm = torch.norm(output, 2, 1)  # output shape (batch, seq, dim)
    var_loss = -torch.sum(norm) / n_element
    var_loss *= args.loss_reg_weight

    loss = mse_loss + cont_loss + var_loss

    # debugging code
    global loss_i
    if loss_i == 1000:
        logging.debug('(custom loss) mse %.5f, cont %.5f, var %.5f'
                      % (mse_loss.item(), cont_loss.item(), var_loss.item()))
        loss_i = 0
    loss_i += 1

    return loss


def train_iter_seq2seq(args, epoch, in_text, in_lengths, target_poses, net, optim):
    # zero gradients
    optim.zero_grad()

    # generation
    outputs = net(in_text, in_lengths, target_poses, None)

    # loss
    loss = custom_loss(outputs, target_poses, args, epoch)
    loss.backward()

    # optimize
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
    optim.step()

    return {'loss': loss.item()}
