import pprint
import time
from pathlib import Path
import sys

[sys.path.append(i) for i in ['.', '..']]

import matplotlib
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model import speech2gesture, vocab
from model.embedding_net import EmbeddingNet
from model.seq2seq_net import Seq2SeqNet
from train_eval.train_gan import train_iter_gan
from train_eval.train_joint_embed import train_iter_embed, eval_embed
from train_eval.train_seq2seq import train_iter_seq2seq
from train_eval.train_speech2gesture import train_iter_speech2gesture
from utils.average_meter import AverageMeter
from utils.data_utils import convert_dir_vec_to_pose
from utils.vocab_utils import build_vocab

matplotlib.use('Agg')  # we don't use interactive GUI

from config.parse_args import parse_args
from model.embedding_space_evaluator import EmbeddingSpaceEvaluator
from model.multimodal_context_net import PoseGenerator, ConvDiscriminator

from torch import optim

from data_loader.lmdb_data_loader import *
import utils.train_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_model(args, lang_model, speaker_model, pose_dim, _device):
    # init model
    n_frames = args.n_poses
    generator = discriminator = loss_fn = None
    if args.model == 'multimodal_context':
        generator = PoseGenerator(args,
                                  n_words=lang_model.n_words,
                                  word_embed_size=args.wordembed_dim,
                                  word_embeddings=lang_model.word_embedding_weights,
                                  z_obj=speaker_model,
                                  pose_dim=pose_dim).to(_device)
        discriminator = ConvDiscriminator(pose_dim).to(_device)
    elif args.model == 'joint_embedding':
        generator = EmbeddingNet(args, pose_dim, n_frames, lang_model.n_words, args.wordembed_dim,
                                 lang_model.word_embedding_weights, mode='random').to(_device)
    elif args.model == 'gesture_autoencoder':
        generator = EmbeddingNet(args, pose_dim, n_frames, lang_model.n_words, args.wordembed_dim,
                                 lang_model.word_embedding_weights, mode='pose').to(_device)
    elif args.model == 'seq2seq':
        generator = Seq2SeqNet(args, pose_dim, n_frames, lang_model.n_words, args.wordembed_dim,
                               lang_model.word_embedding_weights).to(_device)
        loss_fn = torch.nn.L1Loss()
    elif args.model == 'speech2gesture':
        generator = speech2gesture.Generator(n_frames, pose_dim, args.n_pre_poses).to(_device)
        discriminator = speech2gesture.Discriminator(pose_dim).to(_device)
        loss_fn = torch.nn.L1Loss()

    return generator, discriminator, loss_fn


def train_epochs(args, train_data_loader, test_data_loader, lang_model, pose_dim, speaker_model=None):
    start = time.time()
    loss_meters = [AverageMeter('loss'), AverageMeter('var_loss'), AverageMeter('gen'), AverageMeter('dis'),
                   AverageMeter('KLD'), AverageMeter('DIV_REG')]
    best_val_loss = (1e+10, 0)  # value, epoch

    tb_path = args.name + '_' + str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    tb_writer = SummaryWriter(log_dir=str(Path(args.model_save_path).parent / 'tensorboard_runs' / tb_path))

    # interval params
    print_interval = int(len(train_data_loader) / 5)
    save_sample_result_epoch_interval = 10
    save_model_epoch_interval = 20

    # z type
    if args.z_type == 'speaker':
        pass
    elif args.z_type == 'random':
        speaker_model = 1
    else:
        speaker_model = None

    # init model
    generator, discriminator, loss_fn = init_model(args, lang_model, speaker_model, pose_dim, device)

    # use multi GPUs
    if torch.cuda.device_count() > 1:
        generator = torch.nn.DataParallel(generator)
        if discriminator is not None:
            discriminator = torch.nn.DataParallel(discriminator)

    # prepare an evaluator for FGD
    embed_space_evaluator = None
    if args.eval_net_path and len(args.eval_net_path) > 0:
        embed_space_evaluator = EmbeddingSpaceEvaluator(args, args.eval_net_path, lang_model, device)

    # define optimizers
    gen_optimizer = optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    dis_optimizer = None
    if discriminator is not None:
        dis_optimizer = torch.optim.Adam(discriminator.parameters(),
                                         lr=args.learning_rate * args.discriminator_lr_weight,
                                         betas=(0.5, 0.999))

    # training
    global_iter = 0
    best_values = {}  # best values for all loss metrics
    for epoch in range(args.epochs):
        # evaluate the test set
        val_metrics = evaluate_testset(test_data_loader, generator, loss_fn, embed_space_evaluator, args)

        # write to tensorboard and save best values
        for key in val_metrics.keys():
            tb_writer.add_scalar(key + '/validation', val_metrics[key], global_iter)
            if key not in best_values.keys() or val_metrics[key] < best_values[key][0]:
                best_values[key] = (val_metrics[key], epoch)

        # best?
        if 'frechet' in val_metrics.keys():
            val_loss = val_metrics['frechet']
        else:
            val_loss = val_metrics['loss']
        is_best = val_loss < best_val_loss[0]
        if is_best:
            logging.info('  *** BEST VALIDATION LOSS: {:.3f}'.format(val_loss))
            best_val_loss = (val_loss, epoch)
        else:
            logging.info('  best validation loss so far: {:.3f} at EPOCH {}'.format(best_val_loss[0], best_val_loss[1]))

        # save model
        if is_best or (epoch % save_model_epoch_interval == 0 and epoch > 0):
            dis_state_dict = None
            try:  # multi gpu
                gen_state_dict = generator.module.state_dict()
                if discriminator is not None:
                    dis_state_dict = discriminator.module.state_dict()
            except AttributeError:  # single gpu
                gen_state_dict = generator.state_dict()
                if discriminator is not None:
                    dis_state_dict = discriminator.state_dict()

            if is_best:
                save_name = '{}/{}_checkpoint_best.bin'.format(args.model_save_path, args.name)
            else:
                save_name = '{}/{}_checkpoint_{:03d}.bin'.format(args.model_save_path, args.name, epoch)

            utils.train_utils.save_checkpoint({
                'args': args, 'epoch': epoch, 'lang_model': lang_model, 'speaker_model': speaker_model,
                'pose_dim': pose_dim, 'gen_dict': gen_state_dict,
                'dis_dict': dis_state_dict,
            }, save_name)

        # save sample results
        if args.save_result_video and epoch % save_sample_result_epoch_interval == 0:
            evaluate_sample_and_save_video(
                epoch, args.name, test_data_loader, generator,
                args=args, lang_model=lang_model)

        # train iter
        iter_start_time = time.time()
        for iter_idx, data in enumerate(train_data_loader, 0):
            global_iter += 1
            in_text, text_lengths, in_text_padded, _, target_vec, in_audio, in_spec, aux_info = data
            batch_size = target_vec.size(0)

            in_text = in_text.to(device)
            in_text_padded = in_text_padded.to(device)
            in_audio = in_audio.to(device)
            in_spec = in_spec.to(device)
            target_vec = target_vec.to(device)

            # speaker input
            vid_indices = []
            if speaker_model and isinstance(speaker_model, vocab.Vocab):
                vids = aux_info['vid']
                vid_indices = [speaker_model.word2index[vid] for vid in vids]
                vid_indices = torch.LongTensor(vid_indices).to(device)

            # train
            loss = []
            if args.model == 'multimodal_context':
                loss = train_iter_gan(args, epoch, in_text_padded, in_audio, target_vec, vid_indices,
                                      generator, discriminator,
                                      gen_optimizer, dis_optimizer)
            elif args.model == 'joint_embedding':
                loss = train_iter_embed(args, epoch, in_text_padded, in_audio, target_vec,
                                        generator, gen_optimizer, mode='random')
            elif args.model == 'gesture_autoencoder':
                loss = train_iter_embed(args, epoch, in_text_padded, in_audio, target_vec,
                                        generator, gen_optimizer)
            elif args.model == 'seq2seq':
                loss = train_iter_seq2seq(args, epoch, in_text, text_lengths, target_vec, generator, gen_optimizer)
            elif args.model == 'speech2gesture':
                loss = train_iter_speech2gesture(args, in_spec, target_vec, generator, discriminator,
                                                 gen_optimizer, dis_optimizer, loss_fn)

            # loss values
            for loss_meter in loss_meters:
                name = loss_meter.name
                if name in loss:
                    loss_meter.update(loss[name], batch_size)

            # write to tensorboard
            for key in loss.keys():
                tb_writer.add_scalar(key + '/train', loss[key], global_iter)

            # print training status
            if (iter_idx + 1) % print_interval == 0:
                print_summary = 'EP {} ({:3d}) | {:>8s}, {:.0f} samples/s | '.format(
                    epoch, iter_idx + 1, utils.train_utils.time_since(start),
                    batch_size / (time.time() - iter_start_time))
                for loss_meter in loss_meters:
                    if loss_meter.count > 0:
                        print_summary += '{}: {:.3f}, '.format(loss_meter.name, loss_meter.avg)
                        loss_meter.reset()
                logging.info(print_summary)

            iter_start_time = time.time()

    tb_writer.close()

    # print best losses
    logging.info('--------- best loss values ---------')
    for key in best_values.keys():
        logging.info('{}: {:.3f} at EPOCH {}'.format(key, best_values[key][0], best_values[key][1]))


def evaluate_testset(test_data_loader, generator, loss_fn, embed_space_evaluator, args):
    # to evaluation mode
    generator.train(False)

    if embed_space_evaluator:
        embed_space_evaluator.reset()
    losses = AverageMeter('loss')
    joint_mae = AverageMeter('mae_on_joint')
    accel = AverageMeter('accel')
    start = time.time()

    with torch.no_grad():
        for iter_idx, data in enumerate(test_data_loader, 0):
            in_text, text_lengths, in_text_padded, _, target_vec, in_audio, in_spec, aux_info = data
            batch_size = target_vec.size(0)

            in_text = in_text.to(device)
            in_text_padded = in_text_padded.to(device)
            in_audio = in_audio.to(device)
            in_spec = in_spec.to(device)
            target = target_vec.to(device)

            # speaker input
            speaker_model = utils.train_utils.get_speaker_model(generator)
            if speaker_model:
                vid_indices = [random.choice(list(speaker_model.word2index.values())) for _ in range(batch_size)]
                vid_indices = torch.LongTensor(vid_indices).to(device)
            else:
                vid_indices = None

            pre_seq = target.new_zeros((target.shape[0], target.shape[1], target.shape[2] + 1))
            pre_seq[:, 0:args.n_pre_poses, :-1] = target[:, 0:args.n_pre_poses]
            pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
            pre_seq_partial = pre_seq[:, 0:args.n_pre_poses, :-1]

            if args.model == 'joint_embedding':
                loss, out_dir_vec = eval_embed(in_text_padded, in_audio, pre_seq_partial,
                                               target, generator, mode='speech')
            elif args.model == 'gesture_autoencoder':
                loss, _ = eval_embed(in_text_padded, in_audio, pre_seq_partial, target, generator)
            elif args.model == 'seq2seq':
                out_dir_vec = generator(in_text, text_lengths, target, None)
                loss = loss_fn(out_dir_vec, target)
            elif args.model == 'speech2gesture':
                out_dir_vec = generator(in_spec, pre_seq_partial)
                loss = loss_fn(out_dir_vec, target)
            elif args.model == 'multimodal_context':
                out_dir_vec, *_ = generator(pre_seq, in_text_padded, in_audio, vid_indices)
                loss = F.l1_loss(out_dir_vec, target)
            else:
                assert False

            losses.update(loss.item(), batch_size)

            if args.model != 'gesture_autoencoder':
                if embed_space_evaluator:
                    embed_space_evaluator.push_samples(in_text_padded, in_audio, out_dir_vec, target)

                # calculate MAE of joint coordinates
                out_dir_vec = out_dir_vec.cpu().numpy()
                out_dir_vec += np.array(args.mean_dir_vec).squeeze()
                out_joint_poses = convert_dir_vec_to_pose(out_dir_vec)
                target_vec = target_vec.cpu().numpy()
                target_vec += np.array(args.mean_dir_vec).squeeze()
                target_poses = convert_dir_vec_to_pose(target_vec)

                if out_joint_poses.shape[1] == args.n_poses:
                    diff = out_joint_poses[:, args.n_pre_poses:] - target_poses[:, args.n_pre_poses:]
                else:
                    diff = out_joint_poses - target_poses[:, args.n_pre_poses:]
                mae_val = np.mean(np.absolute(diff))
                joint_mae.update(mae_val, batch_size)

                # accel
                target_acc = np.diff(target_poses, n=2, axis=1)
                out_acc = np.diff(out_joint_poses, n=2, axis=1)
                accel.update(np.mean(np.abs(target_acc - out_acc)), batch_size)

    # back to training mode
    generator.train(True)

    # print
    ret_dict = {'loss': losses.avg, 'joint_mae': joint_mae.avg}
    elapsed_time = time.time() - start
    if embed_space_evaluator and embed_space_evaluator.get_no_of_samples() > 0:
        frechet_dist, feat_dist = embed_space_evaluator.get_scores()
        logging.info(
            '[VAL] loss: {:.3f}, joint mae: {:.5f}, accel diff: {:.5f}, FGD: {:.3f}, feat_D: {:.3f} / {:.1f}s'.format(
                losses.avg, joint_mae.avg, accel.avg, frechet_dist, feat_dist, elapsed_time))
        ret_dict['frechet'] = frechet_dist
        ret_dict['feat_dist'] = feat_dist
    else:
        logging.info('[VAL] loss: {:.3f}, joint mae: {:.3f} / {:.1f}s'.format(
            losses.avg, joint_mae.avg, elapsed_time))

    return ret_dict


def evaluate_sample_and_save_video(epoch, prefix, test_data_loader, generator, args, lang_model,
                                   n_save=None, save_path=None):
    generator.train(False)  # eval mode
    start = time.time()
    if not n_save:
        n_save = 1 if epoch <= 0 else 5

    out_raw = []

    with torch.no_grad():
        for iter_idx, data in enumerate(test_data_loader, 0):
            if iter_idx >= n_save:  # save N samples
                break

            in_text, text_lengths, in_text_padded, _, target_dir_vec, in_audio, in_spec, aux_info = data

            # prepare
            select_index = 0
            if args.model == 'seq2seq':
                in_text = in_text[select_index, :].unsqueeze(0).to(device)
                text_lengths = text_lengths[select_index].unsqueeze(0).to(device)
            in_text_padded = in_text_padded[select_index, :].unsqueeze(0).to(device)
            in_audio = in_audio[select_index, :].unsqueeze(0).to(device)
            in_spec = in_spec[select_index, :, :].unsqueeze(0).to(device)
            target_dir_vec = target_dir_vec[select_index, :, :].unsqueeze(0).to(device)

            input_words = []
            for i in range(in_text_padded.shape[1]):
                word_idx = int(in_text_padded.data[select_index, i])
                if word_idx > 0:
                    input_words.append(lang_model.index2word[word_idx])
            sentence = ' '.join(input_words)

            # speaker input
            speaker_model = utils.train_utils.get_speaker_model(generator)
            if speaker_model:
                vid = aux_info['vid'][select_index]
                # vid_indices = [speaker_model.word2index[vid]]
                vid_indices = [random.choice(list(speaker_model.word2index.values()))]
                vid_indices = torch.LongTensor(vid_indices).to(device)
            else:
                vid_indices = None

            # aux info
            aux_str = '({}, time: {}-{})'.format(
                aux_info['vid'][select_index],
                str(datetime.timedelta(seconds=aux_info['start_time'][select_index].item())),
                str(datetime.timedelta(seconds=aux_info['end_time'][select_index].item())))

            # synthesize
            pre_seq = target_dir_vec.new_zeros((target_dir_vec.shape[0], target_dir_vec.shape[1],
                                                target_dir_vec.shape[2] + 1))
            pre_seq[:, 0:args.n_pre_poses, :-1] = target_dir_vec[:, 0:args.n_pre_poses]
            pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
            pre_seq_partial = pre_seq[:, 0:args.n_pre_poses, :-1]

            if args.model == 'multimodal_context':
                out_dir_vec, *_ = generator(pre_seq, in_text_padded, in_audio, vid_indices)
            elif args.model == 'joint_embedding':
                _, _, _, _, _, _, out_dir_vec = generator(in_text_padded, in_audio, pre_seq_partial, None, 'speech')
            elif args.model == 'gesture_autoencoder':
                _, _, _, _, _, _, out_dir_vec = generator(in_text_padded, in_audio, pre_seq_partial, target_dir_vec,
                                                          variational_encoding=False)
            elif args.model == 'seq2seq':
                out_dir_vec = generator(in_text, text_lengths, target_dir_vec, None)
                # out_poses = torch.cat((pre_poses, out_poses), dim=1)
            elif args.model == 'speech2gesture':
                out_dir_vec = generator(in_spec, pre_seq_partial)

            # to video
            audio_npy = np.squeeze(in_audio.cpu().numpy())
            target_dir_vec = np.squeeze(target_dir_vec.cpu().numpy())
            out_dir_vec = np.squeeze(out_dir_vec.cpu().numpy())

            if save_path is None:
                save_path = args.model_save_path

            mean_data = np.array(args.mean_dir_vec).reshape(-1, 3)
            utils.train_utils.create_video_and_save(
                save_path, epoch, prefix, iter_idx,
                target_dir_vec, out_dir_vec, mean_data,
                sentence, audio=audio_npy, aux_str=aux_str)

            target_dir_vec = target_dir_vec.reshape((target_dir_vec.shape[0], 9, 3))
            out_dir_vec = out_dir_vec.reshape((out_dir_vec.shape[0], 9, 3))
            out_raw.append({
                'sentence': sentence,
                'audio': audio_npy,
                'human_dir_vec': target_dir_vec + mean_data,
                'out_dir_vec': out_dir_vec + mean_data,
                'aux_info': aux_str
            })

    generator.train(True)  # back to training mode
    logging.info('saved sample videos, took {:.1f}s'.format(time.time() - start))

    return out_raw


def main(config):
    args = config['args']

    # random seed
    if args.random_seed >= 0:
        utils.train_utils.set_random_seed(args.random_seed)

    # set logger
    utils.train_utils.set_logger(args.model_save_path, os.path.basename(__file__).replace('.py', '.log'))

    logging.info("PyTorch version: {}".format(torch.__version__))
    logging.info("CUDA version: {}".format(torch.version.cuda))
    logging.info("{} GPUs, default {}".format(torch.cuda.device_count(), device))
    logging.info(pprint.pformat(vars(args)))

    # dataset config
    if args.model == 'seq2seq':
        collate_fn = word_seq_collate_fn
    else:
        collate_fn = default_collate_fn

    # dataset
    mean_dir_vec = np.array(args.mean_dir_vec).reshape(-1, 3)
    train_dataset = SpeechMotionDataset(args.train_data_path[0],
                                        n_poses=args.n_poses,
                                        subdivision_stride=args.subdivision_stride,
                                        pose_resampling_fps=args.motion_resampling_framerate,
                                        mean_dir_vec=mean_dir_vec,
                                        mean_pose=args.mean_pose,
                                        remove_word_timing=(args.input_context == 'text')
                                        )
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=args.loader_workers, pin_memory=True,
                              collate_fn=collate_fn
                              )

    val_dataset = SpeechMotionDataset(args.val_data_path[0],
                                      n_poses=args.n_poses,
                                      subdivision_stride=args.subdivision_stride,
                                      pose_resampling_fps=args.motion_resampling_framerate,
                                      speaker_model=train_dataset.speaker_model,
                                      mean_dir_vec=mean_dir_vec,
                                      mean_pose=args.mean_pose,
                                      remove_word_timing=(args.input_context == 'text')
                                      )
    test_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                             shuffle=False, drop_last=True, num_workers=args.loader_workers, pin_memory=True,
                             collate_fn=collate_fn
                             )

    test_dataset = SpeechMotionDataset(args.test_data_path[0],
                                       n_poses=args.n_poses,
                                       subdivision_stride=args.subdivision_stride,
                                       pose_resampling_fps=args.motion_resampling_framerate,
                                       speaker_model=train_dataset.speaker_model,
                                       mean_dir_vec=mean_dir_vec,
                                       mean_pose=args.mean_pose)

    # build vocab
    vocab_cache_path = os.path.join(os.path.split(args.train_data_path[0])[0], 'vocab_cache.pkl')
    lang_model = build_vocab('words', [train_dataset, val_dataset, test_dataset], vocab_cache_path, args.wordembed_path,
                             args.wordembed_dim)
    train_dataset.set_lang_model(lang_model)
    val_dataset.set_lang_model(lang_model)

    # train
    pose_dim = 27  # 9 x 3
    train_epochs(args, train_loader, test_loader, lang_model,
                 pose_dim=pose_dim, speaker_model=train_dataset.speaker_model)


if __name__ == '__main__':
    _args = parse_args()
    main({'args': _args})
