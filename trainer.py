import os
import pickle
import time
import warnings, logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config
from data_utils import get_loader, eta, user_friendly_time, progress_bar, time_since
from model import Seq2seq

warnings.filterwarnings("ignore")
module_logger = logging.getLogger('paragraph-level')


class Trainer(object):
    def __init__(self, args):
        self.logger = logging.getLogger('paragraph-level')

        # train, dev loader
        print("load train data")
        self.train_loader = get_loader(config.train_src_file,
                                       config.train_trg_file,
                                       config.train_ans_file,
                                       batch_size=config.batch_size,
                                       debug=config.debug,
                                       shuffle=True)
        self.dev_loader = get_loader(config.dev_src_file,
                                     config.dev_trg_file,
                                     config.dev_ans_file,
                                     batch_size=128,
                                     debug=config.debug)

        train_dir = os.path.join(config.file_path + "save", "seq2seq")
        self.model_dir = os.path.join(
            train_dir, "train_%d" % int(time.strftime("%m%d%H%M%S")))
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.model = Seq2seq()
        if config.use_gpu:
            self.model = self.model.to(config.device)

        if len(args.model_path) > 0:
            print("load check point from: {}".format(args.model_path))
            state_dict = torch.load(args.model_path,
                                    map_location="cpu")
            self.model.load_state_dict(state_dict)

        params = self.model.parameters()
        bert_params = self.model.bert_encoder.named_parameters()
        for name, param in bert_params:
            param.requires_grad = False
        base_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.lr = config.lr
        self.optim = optim.SGD(base_params, self.lr, momentum=0.8)
        # self.optim = optim.Adam(params)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def save_model(self, loss, epoch):
        state_dict = self.model.state_dict()
        loss = round(loss, 2)
        model_save_path = os.path.join(
            self.model_dir, str(epoch) + "_" + str(loss))
        torch.save(state_dict, model_save_path)

    def train(self):
        batch_num = len(self.train_loader)
        best_loss = 1e10
        for epoch in range(1, config.num_epochs + 1):
            self.model.train()
            print("epoch {}/{} :".format(epoch, config.num_epochs), end="\r")
            start = time.time()
            # halving the learning rate after epoch 8
            if epoch >= 8 and epoch % 2 == 0:
                state_dict = self.optim.state_dict()
                for param_group in state_dict["param_groups"]:
                    param_group["lr"] *= 0.5
                self.optim.load_state_dict(state_dict)

            for batch_idx, train_data in enumerate(self.train_loader, start=1):
                batch_loss = self.step(train_data)

                self.model.zero_grad()
                batch_loss.backward()
                # gradient clipping
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         config.max_grad_norm)

                self.optim.step()
                batch_loss = batch_loss.detach().item()
                msg = "{}/{} {} - ETA : {} - loss : {:.4f}" \
                    .format(batch_idx, batch_num, progress_bar(batch_idx, batch_num),
                            eta(start, batch_idx, batch_num), batch_loss)
                print(msg, end="\r")

            val_loss = self.evaluate(msg)
            if val_loss <= best_loss:
                best_loss = val_loss
                self.save_model(val_loss, epoch)

            print("Epoch {} took {} - final loss : {:.4f} - val loss :{:.4f}"
                  .format(epoch, user_friendly_time(time_since(start)), batch_loss, val_loss))

    def step(self, train_data):
        src_seq, src_mask, trg_seq, trg_mask, tag_seq = train_data
        src_len = torch.sum(src_mask, 1)

        if config.use_gpu:
            src_seq = src_seq.to(config.device)
            src_mask = src_mask.to(config.device)
            trg_seq = trg_seq.to(config.device)
            trg_mask = trg_mask.to(config.device)
            tag_seq = tag_seq.to(config.device)
            src_len = src_len.to(config.device)

        eos_trg = trg_seq[:, 1:]

        if config.use_pointer:
            eos_trg = trg_seq[:, 1:]

        logits = self.model(src_seq, src_mask, tag_seq, trg_seq, trg_mask)

        batch_size, nsteps, _ = logits.size()
        preds = logits.view(batch_size * nsteps, -1)
        targets = eos_trg.contiguous().view(-1)
        loss = self.criterion(preds, targets)

        return loss

    def evaluate(self, msg):
        self.model.eval()
        num_val_batches = len(self.dev_loader)
        val_losses = []
        for i, val_data in enumerate(self.dev_loader, start=1):
            with torch.no_grad():
                val_batch_loss = self.step(val_data)
                val_losses.append(val_batch_loss.item())
                msg2 = "{} => Evaluating :{}/{}".format(
                    msg, i, num_val_batches)
                print(msg2, end="\r")

        val_loss = np.mean(val_losses)

        return val_loss
