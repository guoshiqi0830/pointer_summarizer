from __future__ import unicode_literals, print_function, division

import os
import time
import argparse
import re

import tensorflow as tf
import torch
from model import Model
from torch.nn.utils import clip_grad_norm_

from torch.optim import Adagrad
from torch.optim import Adam

from data_util import config
from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util.utils import calc_running_avg_loss
from data_util.utils import debug
from train_util import get_input_from_batch, get_output_from_batch

use_cuda = config.use_gpu and torch.cuda.is_available()

class Train(object):
    def __init__(self, model_file_path=None):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.train_data_path, self.vocab, mode='train',
                               batch_size=config.batch_size, single_pass=False)
        time.sleep(15)

        if not model_file_path:
          train_dir = os.path.join(config.log_root, 'train_%d' % (int(time.time())))
          if not os.path.exists(train_dir):
              os.mkdir(train_dir)
        else:
          train_dir = re.sub('/model/model.*','',model_file_path)
          
        self.model_dir = os.path.join(train_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.summary_writer = tf.summary.create_file_writer(train_dir)

    def save_model(self, running_avg_loss, iter):
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        model_save_path = os.path.join(self.model_dir, 'model_%d_%d' % (iter, int(time.time())))
        torch.save(state, model_save_path)

    def setup_train(self, model_file_path=None):
        self.model = Model(model_file_path)

        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())
        initial_lr = config.lr_coverage if config.is_coverage else config.lr
        self.optimizer = Adagrad(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)
        # self.optimizer = Adam(params, lr=initial_lr, betas=config.adam_betas, eps=config.adam_eps, 
                              # weight_decay=config.adam_weight_decay)
        start_iter, start_loss = 0, 0

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if not config.is_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                if use_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()

        return start_iter, start_loss

    def f(self, x, alpha):
        # # 1 - x ** alpha
        # k = utils.EPOCH / (utils.MAX_EPOCH / 2) - 1
        # return k * x + (1 - k)/2
        return 1 - x ** alpha

    def get_loss_mask(self,src,tgt,absts,alpha=config.alpha):
        loss_mask = []
        for i in range(len(src)):
          
            # debug('src[i]',src[i])
            # debug('tgt[i]',src[i])
            # cnt = 0
            # tgt_i = [t for t in tgt[i] if t != 1]
            # src_i = set([s for s in src[i] if s != 1])
            # debug('src_i',src_i)
            # m = [t for t in tgt_i if t not in src_i ]
            # # for token in tgt_i:
            # #     if token not in src_i:
            # #         cnt += 1
            # cnt = len(m)
            # abst = round(cnt / len(tgt_i),4)
            abst = absts[i]
            loss_factor = self.f(abst, alpha)
            loss_mask.append(loss_factor)
        return torch.Tensor(loss_mask).cuda()

    def train_one_batch(self, batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)

        self.optimizer.zero_grad()

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state(encoder_hidden)

        # debug(batch.original_articles[0])
        # debug(batch.original_abstracts[0])
        loss_mask = self.get_loss_mask(enc_batch, dec_batch, batch.absts)
        debug('loss_mask',loss_mask)
        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab,
                                                                           coverage, di)
            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)

            debug('enc_batch',enc_batch.size())
            debug('dec_batch',dec_batch.size())
            debug('target',target.size())
            debug('final_dist', final_dist.size())
            debug('gold_probs',gold_probs.size())
            debug('step_loss',step_loss.size())

            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage
                
            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            debug('step_loss_before',step_loss)
            if config.loss_mask:
                step_loss = step_loss * loss_mask
            debug('step_loss_after',step_loss)
            step_losses.append(step_loss)

            if config.DEBUG:
              break

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses/dec_lens_var
        loss = torch.mean(batch_avg_loss)

        loss.backward()

        self.norm = clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()

        return loss.item()

    def trainIters(self, n_iters, model_file_path=None):
        iter, running_avg_loss = self.setup_train(model_file_path)
        start = time.time()
        while iter < n_iters:
            batch = self.batcher.next_batch()
            loss = self.train_one_batch(batch)

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            iter += 1

            if iter % 100 == 0:
                self.summary_writer.flush()
            print_interval = 100
            if iter % print_interval == 0:
                print('steps %d, seconds for %d batch: %.2f , loss: %f' % (iter, print_interval,
                                                                           time.time() - start, loss))
                start = time.time()
            if iter % 5000 == 0:
                self.save_model(running_avg_loss, iter)
            
            if config.DEBUG:
              break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-m",
                        dest="model_file_path", 
                        required=False,
                        default=None,
                        help="Model file for retraining (default: None).")
    args = parser.parse_args()
    
    train_processor = Train(args.model_file_path)
    train_processor.trainIters(config.max_iterations, args.model_file_path)
