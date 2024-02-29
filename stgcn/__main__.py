import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import pdb
import sys
import cv2
import yaml
import torch
import random
import importlib
import faulthandler
import numpy as np
import torch.nn as nn
from collections import OrderedDict
faulthandler.enable()
import utils
from modules.sync_batchnorm import convert_model
from tqdm import tqdm
from slr_eval.wer_calculation import evaluate
from dataloader_video import ph14, CSLFeeder, ph14t

class Processor():
    def __init__(self, cfg):
        self.cfg = cfg
        self.use_ddp = self.cfg['use_ddp']
        if self.use_ddp:
            self.DDP = utils.DistributedDataParallel()
            self.is_master = (self.DDP.rank == 0)
            self.recoder = utils.Recorder(self.cfg['work_dir'], self.cfg['use_tb'], self.is_master)
            self.device = self.DDP.local_rank
            self.rng = utils.RandomState(seed=self.DDP.rank)
        else:
            self.recoder = utils.Recorder(self.cfg['work_dir'], self.cfg['use_tb'])
            self.device = self.cfg['device']
            self.is_master = True
            self.rng = utils.RandomState(seed=self.cfg['random_seed'])
        
        if self.is_master:
        # create work dir
            if not os.path.exists(self.cfg['work_dir']):
                os.makedirs(self.cfg['work_dir'])
            # save configuration
            with open('{}/config.yaml'.format(self.cfg['work_dir']), 'w') as f:
                yaml.dump(self.cfg, f)
        
        self.dataset, self.data_loader, self.gloss_dict = self.load_data()

        self.cfg['model_args']['num_classes'] = len(self.gloss_dict) + 1
        self.min_wer = 1000
        self.model, self.optimizer = self.loading()
        if self.is_master:
            utils.pack_code("./", cfg['work_dir'])

    def start(self):
        if self.cfg['phase'] == 'train':
            self.recoder.print_log('Parameters:\n{}\n'.format(str(self.cfg)))
            for epoch in range(self.cfg['optimizer_args']['start_epoch'], self.cfg['num_epoch']):
                save_model = epoch % self.cfg['save_interval'] == 0
                eval_model = epoch % self.cfg['eval_interval'] == 0
                # train end2end model
                loss_value = self.train_epoch(self.data_loader['train'], epoch, epoch >= self.cfg['cce_activate_epoch'])
                mean_loss = sum(loss_value) / len(loss_value)
                self.recoder.add_scalar('loss/train_epoch', mean_loss, epoch)
                if eval_model and self.is_master:
                    dev_wer = self.eval(self.data_loader['dev'], 'dev', epoch, self.cfg['evaluate_tool'])
                    self.recoder.print_log("Dev WER: {:05.2f}%".format(dev_wer))
                    self.recoder.add_scalar(tag='WER/dev', scalar_value = dev_wer, step = epoch)
                    if dev_wer < self.min_wer:
                        save_model = True
                if save_model and self.is_master:
                    model_path = "{}dev_{:05.2f}_epoch{}_model.pt".format(self.cfg['work_dir'], dev_wer, epoch)
                    self.save_model(epoch, model_path)
                    if dev_wer < self.min_wer:
                        self.min_wer = dev_wer
        
        elif self.cfg['phase'] == 'test':
            dev_wer = self.eval(self.data_loader['test'], 'test', 0, self.cfg['evaluate_tool'])
        else:
            raise NameError

                
    def train_epoch(self, loader, epoch_idx, CCE_activate):
        self.model.train()
        loss_value = []
        clr = [group['lr'] for group in self.optimizer.optimizer.param_groups]
        
        if self.use_ddp:
            loader.sampler.set_epoch(epoch_idx)
            
        for batch_idx, data in enumerate(tqdm(loader)):
            vid = data[0].to(self.device)
            vlen = data[1].to(self.device)
            ske = data[2].to(self.device)
            label = data[3].to(self.device)
            llen = data[4].to(self.device)
            ret_dict = self.model(vid, ske, vlen)
            loss = self.model.criterion_calculation(ret_dict, label, llen, CCE_activate)
            
            if np.isinf(loss.item()) or np.isnan(loss.item()):
                self.optimizer.zero_grad()
                self.optimizer.step()
                continue
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value.append(loss.item())
            if batch_idx % self.cfg['log_interval'] == 0:
                self.recoder.print_log(
                    '\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f}  lr:{:.6f}'
                        .format(epoch_idx, batch_idx, len(loader), loss.item(), clr[0]))
        self.optimizer.scheduler.step()
        self.recoder.print_log('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))
        return loss_value
    
    def eval(self, loader, mode, epoch, evaluate_tool="python"):
        self.model.eval()
        total_sent = []
        total_info = []
        total_conv_sent = []
        for batch_idx, data in enumerate(tqdm(loader)):
            vid = data[0].to(self.device)
            vlen = data[1].to(self.device)
            ske = data[2].to(self.device)

            with torch.no_grad():
                ret_dict = self.model(vid, ske, vlen)
            total_info += [file_name.split("|")[0] for file_name in data[-1]]
            total_sent += ret_dict['recognized_sents']
            total_conv_sent += ret_dict['conv_sents']
        try:
            python_eval = True if evaluate_tool == "python" else False
            self.write2file(self.cfg['work_dir'] + "output-hypothesis-{}.ctm".format(mode), total_info, total_sent)
            self.write2file(self.cfg['work_dir'] + "output-hypothesis-{}-conv.ctm".format(mode), total_info,
                    total_conv_sent)
            conv_ret = evaluate(
                prefix=self.cfg['work_dir'], mode=mode, output_file="output-hypothesis-{}-conv.ctm".format(mode),
                evaluate_dir=self.cfg['dataset_info']['evaluation_dir'],
                evaluate_prefix=self.cfg['dataset_info']['evaluation_prefix'],
                output_dir="epoch_{}_result/".format(epoch),
                python_evaluate=python_eval,
                dataset=self.cfg['dataset']
            )
            lstm_ret = evaluate(
                prefix=self.cfg['work_dir'], mode=mode, output_file="output-hypothesis-{}.ctm".format(mode),
                evaluate_dir=self.cfg['dataset_info']['evaluation_dir'],
                evaluate_prefix=self.cfg['dataset_info']['evaluation_prefix'],
                output_dir="epoch_{}_result/".format(epoch),
                python_evaluate=python_eval,
                triplet=True,
                dataset=self.cfg['dataset']
            )
        except:
            print("Unexpected error:", sys.exc_info()[0])
            lstm_ret = 100.0
        finally:
            pass
        self.recoder.print_log(f"Eval Epoch {epoch}, {mode} {lstm_ret: 2.2f}%")
        return lstm_ret
    
    def write2file(self, path, info, output):
        filereader = open(path, "w")
        for sample_idx, sample in enumerate(output):
            for word_idx, word in enumerate(sample):
                filereader.writelines(
                    "{} 1 {:.2f} {:.2f} {}\n".format(info[sample_idx],
                                                    word_idx * 1.0 / 100,
                                                    (word_idx + 1) * 1.0 / 100,
                                                    word[0]))
    def save_model(self, epoch, save_path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.optimizer.scheduler.state_dict(),
            'rng_state': self.rng.save_rng_state(),
            'min_wer': self.min_wer
        }, save_path)

    def loading(self):
        self.recoder.print_log("Loading model")
        model_class = import_class(self.cfg['model'])
        model = model_class(
            **self.cfg['model_args'],
            gloss_dict=self.gloss_dict,
            loss_weights=self.cfg['loss_weights'],
        ).to(self.device)

        # only works well in single machine
        if self.cfg.get('load_checkpoints', None):
            state_dict = torch.load(self.cfg['load_checkpoints'])
            weights = self.modified_weights(state_dict['model_state_dict'], False)
            model.load_state_dict(weights, strict=True)
            self.recoder.print_log("load model :{}".format(self.cfg['load_checkpoints']))

        if self.use_ddp:
            self.DDP.model_to_device(model)
        optimizer = utils.Optimizer(model, self.cfg['optimizer_args'])

        if self.cfg.get('load_checkpoints', None):
            self.load_checkpoint_weights(optimizer)
            
        self.recoder.print_log("Loading model finished.")
        return model, optimizer

    @staticmethod
    def modified_weights(state_dict, modified=False):
        state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
        if not modified:
            return state_dict
        modified_dict = dict()
        return modified_dict

    def load_checkpoint_weights(self, optimizer):
        state_dict = torch.load(self.cfg['load_checkpoints'])

        if len(torch.cuda.get_rng_state_all()) == len(state_dict['rng_state']['cuda']):
            self.recoder.print_log("Loading random seeds...")
            self.rng.set_rng_state(state_dict['rng_state'])
        if "optimizer_state_dict" in state_dict.keys():
            self.recoder.print_log("Loading optimizer parameters...")
            optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            optimizer.to(self.device)
        if "scheduler_state_dict" in state_dict.keys():
            self.recoder.print_log("Loading scheduler parameters...")
            optimizer.scheduler.load_state_dict(state_dict["scheduler_state_dict"])

        self.cfg['optimizer_args']['start_epoch'] = state_dict["epoch"] + 1
        self.min_wer = state_dict.get('min_wer', self.min_wer)
        self.recoder.print_log(f"Resuming from checkpoint: epoch {self.cfg['optimizer_args']['start_epoch']}")

    def load_data(self):

        self.recoder.print_log("Loading data")
        dataset = {}
        loader = {}
        data_name = self.cfg['dataset']
        self.cfg['dataset_info'] = self.cfg['_'.join([data_name, 'info'])]
        gloss_dict = np.load(self.cfg['dataset_info']['dict_path'], allow_pickle=True).item()
        
        # PHOENIX-2014
        if data_name == 'phoenix14':
            def build_dataloader(dataset, mode, train_flag):
                if train_flag and self.use_ddp:
                    dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
                    loader = torch.utils.data.DataLoader(
                        dataset,
                        sampler = dist_sampler,
                        batch_size=self.cfg['batch_size'] if mode == "train" else self.cfg['test_batch_size'],
                        num_workers=self.cfg['num_worker'],  # if train_flag else 0
                        collate_fn= ph14.collate_fn,
                    )
                else:
                    loader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=self.cfg['batch_size'] if mode == "train" else self.cfg['test_batch_size'],
                        shuffle=train_flag,
                        drop_last=train_flag,
                        num_workers=self.cfg['num_worker'],  # if train_flag else 0
                        collate_fn= ph14.collate_fn,
                    )
                return loader
            
            dataset_list = zip(["train", "dev", "test"], [True, False, False])
            
            for idx, (mode, train_flag) in enumerate(dataset_list):
                arg = self.cfg['feeder_args']
                arg["prefix"] = self.cfg['dataset_info']['dataset_root']
                arg["mode"] = mode.split("_")[0]
                arg["transform_mode"] = train_flag
                dataset[mode] = ph14(gloss_dict=gloss_dict, **arg)
                loader[mode] = build_dataloader(dataset[mode], mode, train_flag)
            
            return dataset, loader, gloss_dict

        if data_name == 'CSL':
            def build_dataloader(dataset, mode, train_flag):
                return torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.cfg['batch_size'] if mode == "train" else self.cfg['test_batch_size'],
                    shuffle=train_flag,
                    drop_last=train_flag,
                    num_workers=self.cfg['num_worker'],  # if train_flag else 0
                    collate_fn= CSLFeeder.collate_fn,
                )
            dataset_list = zip(["train", "dev"], [True, False])
            for idx, (mode, train_flag) in enumerate(dataset_list):
                arg = self.cfg['feeder_args']
                arg["prefix"] = self.cfg['dataset_info']['dataset_root']
                arg["mode"] = mode.split("_")[0]
                arg["transform_mode"] = train_flag
                dataset[mode] = CSLFeeder(gloss_dict=gloss_dict, **arg)
                loader[mode] = build_dataloader(dataset[mode], mode, train_flag)
                
            return dataset, loader, gloss_dict
        if data_name == 'PHOENIX-2014-T':
            def build_dataloader(dataset, mode, train_flag):
                if train_flag and self.use_ddp:
                    dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
                    loader = torch.utils.data.DataLoader(
                        dataset,
                        sampler = dist_sampler,
                        batch_size=self.cfg['batch_size'] if mode == "train" else self.cfg['test_batch_size'],
                        num_workers=self.cfg['num_worker'],  # if train_flag else 0
                        collate_fn= ph14t.collate_fn,
                    )
                else:
                    loader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=self.cfg['batch_size'] if mode == "train" else self.cfg['test_batch_size'],
                        shuffle=train_flag,
                        drop_last=train_flag,
                        num_workers=self.cfg['num_worker'],  # if train_flag else 0
                        collate_fn= ph14t.collate_fn,
                    )
                return loader
            
            dataset_list = zip(["train-complex-annotation", "dev", "test"], [True, False, False])
            
            for idx, (mode, train_flag) in enumerate(dataset_list):
                arg = {}
                arg["prefix"] = self.cfg['dataset_info']['dataset_root']
                arg["mode"] = mode
                arg["transform_mode"] = train_flag
                mode_key = mode.split("-")[0]
                dataset[mode_key] = ph14t(gloss_dict=gloss_dict, **arg)
                loader[mode_key] = build_dataloader(dataset[mode_key], mode_key, train_flag)
            
            return dataset, loader, gloss_dict

        self.recoder.print_log("Loading data finished.")





def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod


if __name__ == '__main__':
    sparser = utils.get_parser('SLR')
    p = sparser.parse_args()
    assert (p.config is not None)
    with open(p.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    processor = Processor(cfg)
    processor.start()