from os.path import join as pj
from copy import deepcopy
import joblib, copy
import pickle
import PIL.Image
import PIL.Image as Im
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

CLIP_LIM = 2.

def truncate(dlatents, dlat_avg, truncation_psi=0.7, maxlayer=8):
#     dlatent_avg = tf.get_default_session().run(Gs.own_vars["dlatent_avg"])
    layer_idx = np.arange(18)[np.newaxis, :, np.newaxis]
    ones = np.ones(layer_idx.shape, dtype=np.float32)
    coefs = tf.where(layer_idx < maxlayer, truncation_psi * ones, ones)
    return tf.get_default_session().run(tflib.lerp(dlat_avg, dlatents, coefs))

def truncate_fancy(dlat, dlat_avg, truncation_psi=0.7, minlayer=0, maxlayer=8, do_clip=False):
    layer_idx = np.arange(model_scale)[np.newaxis, :, np.newaxis]
    ones = np.ones(layer_idx.shape, dtype=np.float32)
    coefs = np.where(layer_idx < maxlayer, truncation_psi * ones, ones)
    if minlayer > 0:
        coefs[0, :minlayer, :] = ones[0, :minlayer, :]
    if do_clip:
        return tflib.lerp_clip(dlat_avg, dlat, coefs).eval()
    else:
        return tflib.lerp(dlat_avg, dlat, coefs)
    
def truncate_multi(dlat, dlat_avg, truncation_psi=0.7, layer_indices=range(8), do_clip=False):
    coefs = np.ones((18, 512), dtype=np.float32)
    for li in layer_indices:
        coefs[li, :] = truncation_psi
        
    if do_clip:
        return tflib.lerp_clip(dlat_avg, dlat, coefs).eval()
    else:
        return tflib.lerp(dlat_avg, dlat, coefs)



def generate_image(latent_vector, imsize=512):
    latent_vector = latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(latent_vector)
    img_array = generator.generate_images()[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    return img.resize((imsize, imsize))

def move_and_show(latent_vector, direction, coeffs, layer_indices=range(8)):
    fig,ax = plt.subplots(1, len(coeffs), figsize=(15, 10), dpi=80)
    for i, coeff in enumerate(coeffs):
        new_latent_vector = latent_vector.copy()
        new_latent_vector[layer_indices] = (latent_vector + coeff*direction)[layer_indices]
        ax[i].imshow(generate_image(new_latent_vector))
        ax[i].set_title('Coeff: %0.1f' % coeff)
    [x.axis('off') for x in ax]
    plt.show()
    
def show_grid(dlats, tr_psi=0.8, tr_dlat=None):
    fig,ax = plt.subplots(1, dlats.shape[0], figsize=(15, 10), dpi=80)
    for i in range(dlats.shape[0]):
        newdlat = dlats[i].copy()
        newdlat = truncate(newdlat, tr_dlat, tr_psi, maxlayer=18)
        ax[i].imshow(generate_image(newdlat))
    [x.axis('off') for x in ax]
    plt.show()
    
def lerp_dir_dlats(dlat, direc, coeffs, layer_indices=range(8)):
    dlats = []
    for i, coeff in enumerate(coeffs):
        dlat_new = copy.deepcopy(dlat)
        dlat_new[layer_indices] = (dlat + coeff*direc)[layer_indices]
        dlats.append(dlat_new)
        
    return dlats


def lerp_dir_dlats_clip(dlat: np.ndarray, direc: np.ndarray, coeffs: np.ndarray, clip_lim=CLIP_LIM, layers=range(8)):
    decay_start = 0.8
    decay_lim = CLIP_LIM * decay_start
    dlat_orig = deepcopy(dlat)
    #coeffs_srt = deepcopy(coeffs)
    #coeffs_srt.sort()
    coef_min = min(coeffs)
    coef_max = max(coeffs)

    dlat_max = deepcopy(dlat)
    dlat_max[layers] = (dlat + coef_max*direc)[layers]

    decay_coeffs = np.where(dlat_max > CLIP_LIM, dlat_max - CLIP_LIM, 0)

    dlats = []
    for i, coeff in enumerate(coeffs):
        dlat_new = deepcopy(dlat)
        dlat_new[layers] = (dlat + coeff*direc)[layers]
        dlat_new[layers] = np.where(dlat_new[layers] > decay_lim, dlat_new[layers] * CLIP_LIM / dlat_max, dlat_new)
        dlats.append(dlat_new)
        
    return dlats


def lerp_dir(latent_vector, direction, coeffs, layer_indices=range(8)):
    imseq = []
    for i, coeff in enumerate(coeffs):
        new_latent_vector = latent_vector.copy()
        new_latent_vector[layer_indices] = (latent_vector + coeff*direction)[layer_indices]
        img = generate_image(new_latent_vector, 512)
        imseq.append(img)
        
    return imseq


def lerp_dir_incrementally(dlat, direc, steps_num=10, step_size=0.1, psi_tr=0.7, dlat_tr=None, layer_indices=range(8)):
    dlats = []
    dlats.append(copy.deepcopy(dlat))
    for i in range(steps_num):
        dlat_prev = copy.deepcopy(dlats[-1])
        dlat_new = copy.deepcopy(dlat_prev) 
        dlat_new[layer_indices] = (dlat_prev + direc * step_size)[layer_indices]
        if not float(psi_tr) == 1. and dlat_tr is not None:
            dlat_new[layer_indices] = truncate(np.expand_dims(dlat_new, axis=0)
                                               , dlat_avg=np.expand_dims(dlat_tr, axis=0)
                                               , truncation_psi=psi_tr)[0][layer_indices]
        dlats.append(dlat_new)
        
    return dlats

def gen_img(latent_vector, generator, imsize=300):
    latent_vector = latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(latent_vector)
    img_array = generator.generate_images()[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    return img.resize((imsize, imsize))

def show_grid_i(imgs):
    fig,ax = plt.subplots(1, len(imgs), figsize=(30, 25), dpi=120)
    for i in range(len(imgs)):
        ax[i].imshow(imgs[i])
    [x.axis('off') for x in ax]
    plt.show()
    
def show_grid(dlats, tr_psi=0.8, tr_dlat=None, do_show=True, imsize=300):
    imgs = []
    for i in range(dlats.shape[0]):
        newdlat = copy.deepcopy(dlats[i])
        newdlat = truncate(newdlat, tr_dlat, tr_psi, maxlayer=8)
        imgs.append(gen_img(newdlat, imsize))
        
    if do_show:    
        fig,ax = plt.subplots(1, len(imgs), figsize=(15, 10), dpi=80)
        for img in imgs:
            ax[i].imshow(img)
        [x.axis('off') for x in ax]
        plt.show()
    
    return imgs
    
def lerp_dir_dlats(dlat, direc, coeffs, layer_indices=range(8)):
    dlats = []
    for i, coeff in enumerate(coeffs):
        dlat_new = copy.deepcopy(dlat)
        dlat_new[layer_indices] = (dlat + coeff*direc)[layer_indices]
        dlats.append(dlat_new)
        
    return dlats



def lerp_dir_incremental(dlat, direc, steps_num=10, step_size=0.1, layer_indices=range(8)):
    dir_shift = direc * step_size
    dlats = []
    dlats.append(copy.deepcopy(dlat))
    for i in range(steps_num):
        dlat_prev = copy.deepcopy(dlats[-1])
        dlat_new = copy.deepcopy(dlat_prev)
        dlat_new[layer_indices] = (dlat_prev + dir_shift)[layer_indices]
        dlat_new = np.where(dlat_new < clip_thresh, dlat_new, dlat_prev + (clip_thresh - dlat_prev)*0.8)
        dlat_new = np.where(dlat_new > -clip_thresh, dlat_new, dlat_prev + (-clip_thresh - dlat_prev)*0.8)    
        dlats.append(dlat_new)
        
    return dlats

def lerp_dir_sym(dla, di, steps_num=4, step_size=0.7, layer_indices = range(8)):
    dlats = lerp_dir_incremental(dla, di, steps_num, step_size, layer_indices)
    dlats_ne = lerp_dir_incremental(dla, di, steps_num, -step_size, layer_indices)
    dlats_ne.reverse()
    dlats = dlats_ne[:-1] + dlats
    imgz = [gen_img(dlat, 300) for dlat in dlats]
    show_grid_i(imgz)


def plot_dlat(dlat, figsize=(10, 7)):
    fig, ax = plt.subplots()
    fig.set_size_inches(figsize)
    ax.plot(dlat, linestyle='', marker='.', alpha=0.6)
    #ax.hlines([-.25, 1.5], xmin=0, xmax=512, colors='k')
    #ax.set_xlim(0, 512)
    #ax.set_ylim(-.5, 1.5)
    plt.show()


def load_pb(model_filepath):
    with tf.gfile.GFile(model_filepath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    tf.import_graph_def(graph_def, name='')
    return graph_def