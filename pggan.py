from __future__ import print_function

import os, sys
from math import floor, ceil

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
import torchvision.transforms as transforms

from network import *
from config import config
import dataloader as dl
import tf_recorder as tensorboard
import utils as utils


class PGGAN:

    def __init__(self, config):
        self.config = config

        if torch.cuda.is_available():
            self.use_cuda = True
            torch.set_default_dtype(torch.float32)
        else:
            self.use_cuda = False
            torch.set_default_dtype(torch.float32)

        print('===========use_cuda========', self.use_cuda)
        self.nz = config.nz
        self.optimizer = config.optimizer
        self.resl = 2       # we start with resolution 2^2 = 4
        self.lr = config.lr
        self.eps_drift = config.eps_drift
        self.smoothing = config.smoothing
        self.max_resl = config.max_resl
        self.trns_tick = config.trns_tick
        self.stab_tick = config.stab_tick
        self.TICK = config.TICK
        self.global_iter = 0
        self.global_tick = 0
        self.kimgs = 0
        self.stack = 0
        self.epoch = 0
        self.fadein = {'gen': None, 'dis': None}
        self.complete = {'gen': 0, 'dis': 0}
        self.phase = 'init'
        self.flag_flush_gen = False
        self.flag_flush_dis = False
        self.flag_add_noise = self.config.flag_add_noise
        self.flag_add_drift = self.config.flag_add_drift

        self.G = Generator(config)
        print('Generator architecture:\n{}'.format(self.G.model))
        self.D = Discriminator(config)
        print('Discriminator architecture:\n{}'.format(self.D.model))
        self.criterion = nn.MSELoss()

        if self.use_cuda:
            self.criterion = self.criterion.cuda()
            torch.cuda.manual_seed(config.random_seed)
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            self.G = nn.DataParallel(self.G).to(self.device)
            self.D = nn.DataParallel(self.D).to(self.device)

        self.renew_everything()

        self.use_tb = config.use_tb
        if self.use_tb:
            self.tb = tensorboard.tf_recorder()

    def resl_scheduler(self):
        if floor(self.resl) != 2:
            self.trns_tick = self.config.trns_tick  # transition in generator
            self.stab_tick = self.config.stab_tick  # stabilize

        self.batchsize = self.loader.batchsize
        delta = 1.0 / (2 * self.trns_tick + 2 * self.stab_tick)  # deltaをどのようにして決めているのか
        d_alpha = 1.0 * self.batchsize / self.trns_tick / self.TICK  # d_alpha をどのようにして決めているのか

        # fade-in layerとはなんなのか ← おそらく解像度をあげる際の中間層
        # question : genとdisが同時に更新対象になることがない理由がよくらかない。
        # 　なぜならないのか計算をする必要性がある
        # reslの更新条件
        if self.fadein['gen'] is not None:
            if self.resl % 1.0 < self.trns_tick * delta:  # どういう意味を持つのか　どういう時に更新をするのか
                self.fadein['gen'].update_alpha(d_alpha)
                self.complete['gen'] = self.fadein['gen'].alpha * 100
                self.phase = 'gtrns'

            elif (self.resl % 1.0 >= self.trns_tick * delta) \
                    and self.resl % 1.0 < (self.trns_tick + self.stab_tick) * delta:
                self.phase = 'gstab'

        if self.fadein['dis'] is not None:
            if self.resl % 1.0 >= (self.trns_tick + self.stab_tick) * delta \
                    and (self.resl % 1.0) < (self.stab_tick + self.trns_tick * 2) * delta:
                self.fadein['dis'].update_alpha(d_alpha)
                self.complete['dis'] = self.fadein['dis'].alpha * 100  # completeがよくわからない
                self.phase = 'dtrns'

            elif self.resl % 1.0 >= (self.stab_tick + self.trns_tick * 2) * delta and self.phase != 'final':
                self.phase = 'dstab'

        prev_kimgs = self.kimgs  # k-imgs
        self.kimgs = self.kimgs + self.batchsize
        if self.kimgs % self.TICK < prev_kimgs % self.TICK:  # 今回の方が通したimgが多い場合
            self.global_tick = self.global_tick + 1

            prev_resl = floor(self.resl)
            self.resl = self.resl + delta  # deltaはreslを更新する幅
            self.resl = max(2, min(10.5, self.resl))  # 2 ~ 10.5

            # Flush Network
            if self.flag_flush_gen and \
                    self.resl % 1.0 >= (self.trns_tick + self.stab_tick) * delta and prev_resl != 2:
                if self.fadein['gen'] is not None:
                    self.fadein['gen'].update_alpha(d_alpha)
                    self.complete['gen'] = self.fadein['gen'].alpha * 100
                self.flag_flush_gen = False
                self.G.module.flush_network()
                print('Generator flushed:\{}'.format(self.G.module.model))
                self.fadein['gen'] = None
                self.complete['gen'] = 0.0
                self.phase = 'dtrns'

            elif self.flag_flush_dis and floor(self.resl) != prev_resl and prev_resl != 2:
                if self.fadein['dis'] is not None:
                    self.fadein['dis'].update_alpha(d_alpha)
                    self.complete['dis'] = self.fadein['dis'].alpha * 100

                self.flag_flush_dis = False
                self.D.module.flush_network()
                print('Discriminator flushed:\n{}'.format(self.D.module.model))
                self.fadein['dis'] = None
                self.complete['dis'] = 0.0
                if floor(self.resl) < self.max_resl and self.phase != 'final':
                    self.phase = 'gtrns'

            # Grow the networks
            # reslの値が更新されていて、max以下の値である場合にgrowする
            if floor(self.resl) != prev_resl and floor(self.resl) < self.max_resl + 1:
                self.lr = self.lr * float(self.config.lr_decay)
                self.G.module.grow_network(floor(self.resl))
                self.D.module.grow_network(floor(self.resl))  # grow_networkでおそらくfadein_blockができるのではないか
                self.renew_everything()
                self.fadein['gen'] = self.G.module.model.fadein_block
                self.fadein['dis'] = self.D.module.model.fadein_block
                self.flag_flush_gen = True
                self.flag_flush_dis = True

            # 最終の場合、状態を変える
            if floor(self.resl) >= self.max_resl and \
                    self.resl % 1.0 >= (self.stab_tick + self.trns_tick * 2) * delta:
                self.phase = 'final'
                self.resl = self.max_resl + (self.stab_tick + self.trns_tick * 2) * delta


    def renew_everything(self):
        self.loader = dl.dataloader(self.config)
        self.loader.renew(min(floor(self.resl), self.max_resl))

        # define tensors
        #     self.z = torch.FloatTensor(self.loader.batchsize, self.nz)
        self.z = torch.randn([self.loader.batchsize, self.nz],
                             requires_grad=True, dtype=torch.float32)
        self.x = torch.randn([self.loader.batchsize, 3, self.loader.imsize, self.loader.imsize],
                             requires_grad=True, dtype=torch.float32)
        self.x_tilde = torch.randn([self.loader.batchsize, 3, self.loader.imsize, self.loader.imsize],
                                   requires_grad=True, dtype=torch.float32)

        self.real_label = torch.ones(self.loader.batchsize, requires_grad=True)
        self.fake_label = torch.zeros(self.loader.batchsize, requires_grad=True)

        # Enable CUDA
        if self.use_cuda:
            self.z = self.z.to(self.device)
            self.x = self.x.to(self.device)
            self.x_tilde = self.x_tilde.to(self.device)
            self.real_label = self.real_label.to(self.device)
            self.fake_label = self.fake_label.to(self.device)
            torch.cuda.manual_seed(config.random_seed)

        if self.use_cuda:
            self.G = self.G.to(self.device)
            self.D = self.D.to(self.device)

        betas = (self.config.beta1, self.config.beta2)
        if self.optimizer == 'adam':
            self.opt_g = Adam(filter(lambda p: p.requires_grad, self.G.parameters()), lr=self.lr, betas=betas,
                              weight_decay=0.0)
            self.opt_d = Adam(filter(lambda p: p.requires_grad, self.D.parameters()), lr=self.lr, betas=betas,
                              weight_decay=0.0)

    def feed_interpolated_input(self, x):
        if (self.phase == 'gtrns') and (floor(self.resl) > 2) and floor(self.resl) <= self.max_resl:
            alpha = self.complete['gen'] / 100.0
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(size=int(pow(2, floor(self.resl) - 1)), interpolation=0),
                transforms.Resize(size=int(pow(2, floor(self.resl))), interpolation=0),
                transforms.ToTensor(),
            ])
            x_low = x.clone().add(1).mul(0.5)  # (x + 1)*0.5
            for i in range(x_low.size(0)):
                x_low[i] = transform(x_low[i]).mul(2).add(-1)
            x = torch.add(x.mul(alpha), x_low.mul(1 - alpha))  # 合体してるんじゃない？
        if self.use_cuda:
            return x.to(self.device)
        else:
            return x

    def add_noise(self, x):
        if not self.flag_add_noise:
            return x

        if hasattr(self, '_d_'):  # なんだこれ
            self._d_ = self._d_ * 0.9 + torch.mean(self.fx_tilde).item() * 0.1
        else:
            self._d_ = 0.0

        strength = 0.2 * max(0, self._d_ - 0.5) ** 2
        z = np.random.randn(*x.size()).astype(np.float32) * strength
        z = torch.tensor(z, requires_grad=True)
        z = z.to(self.device) if self.use_cuda else z
        return x + z

    def train(self):
        # noise for test
        self.z_test = torch.randn([self.loader.batchsize, self.nz], requires_grad=True, dtype=torch.float32)
        if self.use_cuda:
            self.z_test = self.z_test.to(self.device)
        self.z_test.data.resize_(self.loader.batchsize, self.nz).normal_(0.0, 1.0)

        for step in range(0, self.max_resl + 1 + 5):
            for _ in tqdm(range(0, (self.trns_tick * 2 + self.stab_tick * 2) * self.TICK, self.loader.batchsize)):
                print('=============batch_size=============', self.loader.batchsize)
                self.global_iter = self.global_iter + 1
                self.stack = self.stack + self.loader.batchsize  # What is "stack" ??
                if self.stack > ceil(len(self.loader.dataset)):
                    self.epoch = self.epoch + 1
                    self.stack = int(self.stack % (ceil(len(self.loader.dataset))))

                # Resolution scheduler
                self.resl_scheduler()

                # Zero the gradients
                self.G.zero_grad()
                self.D.zero_grad()

                # update discriminator
                self.x.data = self.feed_interpolated_input(self.loader.get_batch())
                if self.flag_add_noise:
                    self.x = self.add_noise(self.x)
                self.z.data.resize_(self.loader.batchsize, self.nz).normal_(0.0, 1.0)
                self.x_tilde = self.G(self.z)  # x_tlide は Gから生成した画像

                self.fx = self.D(self.x)
                self.fx_tilde = self.D(self.x_tilde.detach())

                # Discriminator の Loss関数について調べる　
                loss_d = self.criterion(self.fx, self.real_label) + self.criterion(self.fx_tilde, self.fake_label)

                # Compute gradients adn apply upgrade to parameters
                loss_d.backward()
                self.opt_d.step()

                # Update generator
                fx_tilde = self.D(self.x_tilde)
                loss_g = self.criterion(fx_tilde, self.real_label.detach())

                # Compute gradients adn apply upgrade to parameters
                loss_g.backward()
                self.opt_g.step()

                log_msg = ' [E:{0}][T:{1}][{2:6}/{3:6}]  errD: {4:.4f} | errG: {5:.4f} | [lr:{11:.5f}][cur:{6:.3f}][resl:{7:4}][{8}][{9:.1f}%][{10:.1f}%]'.format(
                    self.epoch, self.global_tick, self.stack, len(self.loader.dataset), loss_d.item(), loss_g.item(), self.resl, int(pow(2,floor(self.resl))), self.phase, self.complete['gen'], self.complete['dis'], self.lr)
                tqdm.write(log_msg)
                self.snapshot('./repo/model')

                # Save the image grid
                if self.global_iter % self.config.save_img_every == 0:
                    x_test = self.G(self.z_test)
                    os.system('mkdir -p repo/save/grid')
                    utils.save_image_grid(x_test.data, 'repo/save/grid/{}_{}_G{}_D{}.jpg'.format(
                        int(self.global_iter / self.config.save_img_every), self.phase, self.complete['gen'],
                        self.complete['dis']))
                    os.system('mkdir -p repo/save/resl_{}'.format(int(floor(self.resl))))
                    utils.save_image_single(x_test.data,
                                            'repo/save/resl_{}/{}_{}_G{}_D{}.jpg'.format(int(floor(self.resl)), int(
                                                self.global_iter / self.config.save_img_every), self.phase,
                                                                                         self.complete['gen'],
                                                                                         self.complete['dis']))

                # Tensorboard visualization
                if self.use_tb:
                    x_test = self.G(self.z_test)
                    self.tb.add_scalar('data/loss_g', loss_g.item(), self.global_iter)
                    self.tb.add_scalar('data/loss_d', loss_d.item(), self.global_iter)
                    self.tb.add_scalar('tick/lr', self.lr, self.global_iter)
                    self.tb.add_scalar('tick/cur_resl', int(pow(2, floor(self.resl))), self.global_iter)
                    self.tb.add_image_grid('grid/x_test', 4,
                                           utils.adjust_dyn_range(x_test.data.float(), [-1, 1], [0, 1]),
                                           self.global_iter)
                    self.tb.add_image_grid('grid/x_tilde', 4,
                                           utils.adjust_dyn_range(self.x_tilde.data.float(), [-1, 1], [0, 1]),
                                           self.global_iter)
                    self.tb.add_image_grid('grid/x_intp', 4,
                                           utils.adjust_dyn_range(self.x.data.float(), [-1, 1], [0, 1]),
                                           self.global_iter)

    def get_state(self, target):
        if target == 'gen':
            state = {
                'resl' : self.resl,
                'state_dict' : self.G.state_dict(),
                'optimizer' : self.opt_g.state_dict(),
            }
            return state
        elif target == 'dis':
            state = {
                'resl' : self.resl,
                'state_dict' : self.D.state_dict(),
                'optimizer' : self.opt_d.state_dict(),
            }
            return state

    def snapshot(self, path):
        if not os.path.exists(path):
            os.system('mkdir -p {}'.format(path))
        # Save every 100 tick if the network is in stab phase
        ndis = 'dis_R{}_T{}.pth.tar'.format(int(floor(self.resl)), self.global_tick)
        ngen = 'gen_R{}_T{}.pth.tar'.format(int(floor(self.resl)), self.global_tick)
        if self.global_tick % 50 == 0:
            if self.phase == 'gstab' or self.phase == 'dstab' or self.phase == 'final':
                save_path = os.path.join(path, ndis)
                if not os.path.exists(save_path):
                    torch.save(self.get_state('dis'), save_path)
                    save_path = os.path.join(path, ngen)
                    torch.save(self.get_state('gen'), save_path)
                    print('[snapshot] model saved @ {}'.format(path))

    def evaluate(self):
        pass

    def test_growth(self):
        self.G.module.grow_network(3)
        self.G.module.flush_network()
        print(self.G.module.model)

# Perform the training of PGGAN
print('Configuration')
for k, v in vars(config).items():
    print('{}: {}'.format(k, v))

torch.backends.cudnn.benchmark = True   # boost the speed
pggan = PGGAN(config)
pggan.train()