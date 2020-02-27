import ipywidgets as wdgts
from ipywidgets import GridspecLayout, HBox, Output
from IPython.display import display
import os, sys, joblib, copy, pickle, PIL.Image as Im, numpy as np, matplotlib.pyplot as plt, cv2, shutil, argparse, pathlib, re
import dnnlib, pretrained_networks, dnnlib.tflib as tflib, utils_stylegan as ust
from training import dataset, misc
from os.path import join as pj
from io import BytesIO
from scipy.stats import wasserstein_distance as wass_dist


proj_presets = {
    'best_p3': dnnlib.EasyDict(initial_learning_rate = 0.02,
                                lr_rampdown_length = 0.55,
                                lr_rampup_length = 0.05,
                                coef_pixel_loss = 0.01,
                                coef_mssim_loss = 1.5,
                                coef_dlat_loss = .3,
                                num_steps = 3000,
                                verbose = True),
    'best_p3_refine': dnnlib.EasyDict(initial_learning_rate = 0.01,
                                      path_mask = './assets/masks/mask_1024.png',
                                      lr_rampdown_length = 0.75,
                                      lr_rampup_length = 0.01,
                                      coef_pixel_loss = 0,
                                      coef_mssim_loss = 1.5,
                                      coef_dlat_loss = 0.05,
                                      num_steps = 3000,
                                      verbose = True),
    'best_p3_refine1': dnnlib.EasyDict(initial_learning_rate = 0.03,
                                      path_mask = './assets/masks/mask_1024.png',
                                      lr_rampdown_length = 0.65,
                                      lr_rampup_length = 0.01,
                                      coef_pixel_loss = 0,
                                      coef_mssim_loss = 1.,
                                      coef_dlat_loss = 0.08,
                                      num_steps = 1000,
                                      verbose = True),
    'best_p3_refine2': dnnlib.EasyDict(initial_learning_rate = 0.03,
                                      path_mask = './assets/masks/mask_1024_enh.png',
                                      lr_rampdown_length = 0.65,
                                      lr_rampup_length = 0.01,
                                      coef_pixel_loss = 0,
                                      coef_mssim_loss = 2.,
                                      coef_dlat_loss = 0.2,
                                      num_steps = 1000,
                                      verbose = True),
    'p1': dnnlib.EasyDict(initial_learning_rate = 0.2,
                          lr_rampdown_length = 0.65,
                          lr_rampup_length = 0.05,
                          coef_pixel_loss = 0,
                          coef_mssim_loss = 1.5,
                          coef_dlat_loss = .05,
                          num_steps = 1000,
                          verbose = True),
    'p2': dnnlib.EasyDict(initial_learning_rate = 0.1,
                          lr_rampdown_length = 0.45,
                          lr_rampup_length = 0.05,
                          coef_pixel_loss = 0.01,
                          coef_mssim_loss = 1.,
                          coef_dlat_loss = .1,
                          num_steps = 1000,
                          verbose = True),
    'p3': dnnlib.EasyDict(initial_learning_rate = 0.1,
                          image_size = 512,
                          path_mask = './assets/masks/mask_1024_softscaled.png',
                          lr_rampdown_length = 0.75,
                          lr_rampup_length = 0.05,
                          coef_pixel_loss = 0.01,
                          coef_mssim_loss = 1.,
                          coef_dlat_loss = .5,
                          num_steps = 1000,
                          verbose = True)
}


def write_video_frame(img, video_out):
    video_frame = img.resize((512, 512))
    video_out.write(cv2.cvtColor(np.array(video_frame).astype('uint8'), cv2.COLOR_RGB2BGR))


def project_image(proj, targets, png_prefix, num_snapshots, out_widget=None, out_widget_dlat=None):
    video_out = cv2.VideoWriter(png_prefix + '_video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 24, (512, 512))
    snapshot_steps = set(proj.num_steps - np.linspace(0, proj.num_steps, num_snapshots, endpoint=False, dtype=int))
    #misc.save_image_grid(np.expand_dims(targets[0], axis=0), png_prefix + '_target.png', drange=[-1 ,1])
    proj.start(targets)
    while proj.get_cur_step() < proj.num_steps:
        proj.step()
        imgout = misc.convert_to_pil_image(misc.create_image_grid(proj.get_images(), None), drange=[-1 ,1]).resize((512, 512))
        if proj.get_cur_step() % 10 == 0 and out_widget is not None:
            if out_widget is not None:
                out_widget.clear_output()
                with out_widget:
                    display(imgout)
            if out_widget_dlat is not None:
                out_widget_dlat.clear_output()
                with out_widget_dlat:
                    plot_dlat_fancy(proj.get_dlatents()[0][:8], proj._dlatent_avg[0, 0], proj._dlatent_std[0, 0])

        write_video_frame(imgout, video_out)
        if proj.get_cur_step() in snapshot_steps:
            misc.save_image_grid(proj.get_images(), png_prefix + '_step%04d.png' % proj.get_cur_step(), drange=[-1 ,1])

    dlats = proj.get_dlatents()
    joblib.dump(dlats, png_prefix + '_dlats.jbl')
    video_out.release()


class WidgetRepo:
    def __init__(self):
        self.isinit = True
        self.lids = range(10)
        self.Gs = None

    def init_Gs(self):
        if self.Gs is None:
            network_pkl = 'gdrive:networks/stylegan2-ffhq-config-f.pkl'
            _, _, self.Gs = pretrained_networks.load_networks(network_pkl)

        return self.Gs

#wrepo = WidgetRepo


def wass_dist_mean(dlat, dlat_vec):
    dists = []
    for dlat1d in dlat:
        dist = wass_dist(dlat_vec, dlat1d)
        dists.append(dist)

    return np.array(dists).mean()


def gather_direc(direcs):
    direc = None
    for sln, sl in wrepo.sliders.items():
        if direc is None:
            direc = sl.value * direcs[sln]
        else:
            direc += sl.value * direcs[sln]

    return direc


def create_sliders(names):
    slily = wdgts.Layout(width='250px')
    res = {}
    for name in names:
        res[name] = wdgts.FloatSlider(value=0., min=-1., max=1., step=0.25, orientation='horizontal', description=name,
                                      layout=slily)
        res[name].observe(on_value_change, names='value')

    return res


def on_value_change(change):
    try:
        newval = change.new
    except:
        newval = change
    direc = gather_direc(wrepo.direcs)
    dlat = ust.lerp_dir_dlats(wrepo.dlat, direc, [wrepo.slider_coef.value], layer_indices=wrepo.lids)
    wrepo.dlat_plot.clear_output()
    with wrepo.dlat_plot:
        plot_dlat(dlat[0][wrepo.lids])

    handle_render({})


def plot_dlat(dlat):
    ax, fig = plt.subplots()
    ax.set_figheight(5)
    ax.set_figwidth(8)
    plt.plot(dlat[wrepo.lids].T, linestyle='', marker='.', alpha=0.2, markersize=5, antialiased=True)
    # plt.hlines([-2, 2.], xmin=0, xmax=512, colors='k')
    plt.xlim(0, 512)
    plt.show()


def plot_dlat_fancy(dlat, dlat_avg, dlat_std, figsize=(8, 5)):
    fig, ax = plt.subplots()
    fig.set_figheight(figsize[1])
    fig.set_figwidth(figsize[0])
    upper_dlat = dlat_avg + dlat_std
    lower_dlat = dlat_avg - dlat_std
    # plt.plot(upper_dlat.T, linestyle='', color='c', marker='.', alpha=0.9, markersize=4,  antialiased=True)
    # plt.plot(lower_dlat.T, linestyle='', color='c', marker='.', alpha=0.9, markersize=4, antialiased=True)
    ax.fill_between(range(512), lower_dlat, upper_dlat, antialiased=True, alpha=0.3)
    plt.plot(dlat[wrepo.lids].T, linestyle='', color='r', marker='.', alpha=0.7, markersize=3, antialiased=True)
    plt.xlim(0, 512)
    plt.show()


def handle_render(obj):
    direc = gather_direc(wrepo.direcs)
    dlats_n = ust.lerp_dir_dlats(wrepo.dlat, direc, [wrepo.slider_coef.value], layer_indices=wrepo.lids)
    images = wrepo.Gs.components.synthesis.run(np.array(dlats_n), **wrepo.Gs_syn_kwargs)
    images_pil = [Im.fromarray(img, 'RGB') for img in images]
    wrepo.image_out.clear_output()
    with wrepo.image_out:
        display(images_pil[0].resize((300, 300)))



def create_ui():
    wrepo.btn_render = wdgts.Button(description='Render')
    wrepo.btn_render.on_click(handle_render)

    wrepo.sliders = create_sliders(list(wrepo.direcs.keys()))
    wrepo.slider_coef = wdgts.FloatSlider(value=0., min=-15., max=15, step=1, orientation='horizontal',
                                               description='Shift')
    wrepo.slider_coef.observe(on_value_change, names='value')

    wrepo.image_init_out = wdgts.Output()
    wrepo.image_out = wdgts.Output()
    wrepo.dlat_plot = wdgts.Output()

    lay = wdgts.GridspecLayout(1, 2)
    lay[0, 0] = wdgts.VBox([wrepo.slider_coef, wrepo.btn_render] + list(wrepo.sliders.values()))
    lay[0, 1] = wdgts.VBox(
        [wdgts.HBox([wrepo.image_init_out, wrepo.image_out],
                    layout=wdgts.Layout(width='650px', height='350px')), wrepo.dlat_plot],
        layout=wdgts.Layout(width='650px'))

    return lay