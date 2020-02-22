# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
import PIL.Image as Im, cv2

from training import misc

#----------------------------------------------------------------------------

class Projector:
    def __init__(self):
        self.num_steps                  = 1000
        self.dlatent_avg_samples        = 10000
        self.img_size                   = 256
        self.initial_learning_rate      = 0.1
        self.lr_rampdown_length         = 0.25
        self.lr_rampup_length           = 0.05
        self.verbose                    = False
        self.path_mask                  = './assets/masks/mask_1024_soft.png'
        self.clone_net                  = False
        self.num_dlats_smpls            = 10


        self.coef_pixel_loss = 0.
        self.coef_dlat_loss = 0.
        self.coef_mssim_loss = 0.

        self._Gs                    = None
        self._minibatch_size        = None
        self._dlatent_avg           = None
        self._dlatent_std           = None
        self._dlas_smpls            = None
        self._noise_vars            = None
        self._dlatents_var          = None
        self._images_expr           = None
        self._target_images_var     = None
        self._lpips                 = None
        self._dist                  = None
        self._losses                = None
        self._loss                  = None
        self._reg_sizes             = None
        self._lrate_in              = None
        self._opt                   = None
        self._opt_step              = None
        self._cur_step              = None
        self._img_mask_rgb_np = None

    def _info(self, *args):
        if self.verbose:
            print('Projector:', *args)


    def gen_dlats_smpls(self, num_smpls, do_tile=False):
        latent_samples = np.random.RandomState(123).randn(num_smpls, *self._Gs.input_shapes[0][1:])
        dlatent_samples = self._Gs.components.mapping.run(latent_samples, None)  # [N, 1, 512]
        return dlatent_samples

    def set_network(self, Gs, minibatch_size=1):
        assert minibatch_size == 1
        self._Gs = Gs
        self._minibatch_size = minibatch_size
        if self._Gs is None:
            return
        if self.clone_net:
            self._Gs = self._Gs.clone()

        # Find dlatent stats.
        self._info('Finding W midpoint and stddev using %d samples...' % self.dlatent_avg_samples)
        latent_samples = np.random.RandomState(123).randn(self.dlatent_avg_samples, *self._Gs.input_shapes[0][1:])
        dlatent_samples = self._Gs.components.mapping.run(latent_samples, None) # [N, 1, 512]
        self._dlatent_avg = np.mean(dlatent_samples, axis=0, keepdims=True) # [1, 18, 512]
        self._dlatent_std = (np.sum((dlatent_samples - self._dlatent_avg) ** 2) / self.dlatent_avg_samples) ** 0.5
        self._info('std = %g' % self._dlatent_std)

        img_mask = Im.open(self.path_mask).convert('RGB')
        self._mask_rgb_np = np.expand_dims(img_mask, axis=0) / 255
        #self._mask_rgb_perc_np = np.reshape(img_mask.resize((self.img_size, self.img_size)), newshape=(1, self.img_size, self.img_size, 3)) / 255

        # Image output graph.
        self._info('Building image output graph...')
        self._dlatents_var = tf.Variable(tf.zeros([self._minibatch_size] + list(self._dlatent_avg.shape[1:])), name='dlatents_var')
        self._images_expr = self._Gs.components.synthesis.get_output_for(self._dlatents_var, randomize_noise=False)
        print(self._dlatents_var)
        proc_images_expr = tf.transpose((self._images_expr + 1) * (255 / 2), (0, 2, 3, 1))
        self._proc_images_masked_expr = proc_images_expr * self._mask_rgb_np

        # Loss graph.
        self._losses = []
        self._info('Building loss graph...')
        self._target_images_var = tf.Variable(tf.zeros(proc_images_expr.shape), name='target_images_var')
        if self._lpips is None:
            self._lpips = misc.load_pkl('http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/vgg16_zhang_perceptual.pkl') # vgg16_zhang_perceptual.pkl

        proc_images_perc = tf.transpose(tf.image.resize_images(self._proc_images_masked_expr, (self.img_size, self.img_size)), (0, 3, 1, 2))
        targ_images_perc = tf.transpose(tf.image.resize_images(self._target_images_var, (self.img_size, self.img_size)), (0, 3, 1, 2))
        self._dist = self._lpips.get_output_for(proc_images_perc, targ_images_perc)

        self._losses.append(tf.reduce_sum(self._dist))
        self._loss = self._losses[-1]

        # Plain pixel loss
        if self.coef_pixel_loss > 0:
            #self._losses.append(tf.losses.mean_squared_error(proc_images_masked_g_expr, targ_images_g_expr))
            self._losses.append(tf.math.reduce_sum(tf.math.abs(self._proc_images_masked_expr - self._target_images_var)))
            self._loss += self.coef_pixel_loss * self._losses[-1]

        if self.coef_mssim_loss > 0:
            proc_images_masked_g_expr = tf.image.rgb_to_grayscale(self._proc_images_masked_expr)
            targ_images_g_expr = tf.image.rgb_to_grayscale(self._target_images_var)
            self._losses.append(tf.math.reduce_sum(1 - tf.image.ssim(proc_images_masked_g_expr, targ_images_g_expr, max_val=255.)))
            self._loss += self.coef_mssim_loss * self._losses[-1]

        # Random dlat penalty
        if self.coef_dlat_loss > 0:
            #self._dlats_smpls = tf.Variable(tf.zeros([self.num_dlats_smpls, 512]), name='rnd_dlats_smpls')
            self._losses.append(tf.math.reduce_mean(tf.math.abs(self._dlatent_avg - self._dlatents_var)))
            self._loss += self.coef_dlat_loss * self._losses[-1]

        # Optimizer.
        self._info('Setting up optimizer...')
        self._lrate_in = tf.placeholder(tf.float32, [], name='lrate_in')
        self._opt = dnnlib.tflib.Optimizer(learning_rate=self._lrate_in)
        self._opt.register_gradients(self._loss, [self._dlatents_var])
        self._opt_step = self._opt.apply_updates()

    def run(self, target_images):
        # Run to completion.
        self.start(target_images)
        while self._cur_step < self.num_steps:
            self.step()

        # Collect results.
        pres = dnnlib.EasyDict()
        pres.dlatents = self.get_dlatents()
        pres.images = self.get_images()
        return pres

    def start(self, target_images):
        assert self._Gs is not None

        # Prepare target images.
        self._info('Preparing target images...')
        target_images = np.asarray(target_images, dtype='float32')
        target_images = (target_images + 1) * (255 / 2)
        target_images_nhwc = np.transpose(target_images, [0, 2, 3, 1])
        target_images_nhwc_masked = target_images_nhwc * self._mask_rgb_np
        # Initialize optimization state.
        self._info('Initializing optimization state...')
        tflib.set_vars({self._target_images_var: target_images_nhwc_masked, self._dlatents_var: np.tile(self._dlatent_avg, [self._minibatch_size, 1, 1])})
        self._opt.reset_optimizer_state()
        self._cur_step = 0

    def step(self):
        assert self._cur_step is not None
        if self._cur_step >= self.num_steps:
            return
        if self._cur_step == 0:
            self._info('Running...')

        # Hyperparameters.
        t = self._cur_step / self.num_steps
        lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
        learning_rate = self.initial_learning_rate * lr_ramp

        # Train.
        feed_dict = {self._lrate_in: learning_rate}
        res_lst = tflib.run([self._opt_step, self._loss] + self._losses, feed_dict)

        # Print status.
        self._cur_step += 1
        if self._cur_step == self.num_steps or self._cur_step % 10 == 0:
            self._info(f'loss: {res_lst[1]}, losses: {res_lst[2:]}')
        if self._cur_step == self.num_steps:
            self._info('Done.')

    def get_cur_step(self):
        return self._cur_step

    def get_dlatents(self):
        return tflib.run(self._dlatents_var)

    def get_images(self):
        return tflib.run(self._images_expr)

#----------------------------------------------------------------------------
