import ipywidgets as wdgts
from ipywidgets import GridspecLayout, HBox, Output
from IPython.display import display
import os, sys, joblib, copy, pickle, PIL.Image as Im, numpy as np, matplotlib.pyplot as plt, cv2, shutil, argparse, pathlib, re
import dnnlib, pretrained_networks, dnnlib.tflib as tflib, utils_stylegan as ust
from training import dataset, misc
from os.path import join as pj
from io import BytesIO
from scipy.stats import wasserstein_distance as wass_dist


def write_video_frame(img, video_out):
    video_frame = img.resize((512, 512))
    video_out.write(cv2.cvtColor(np.array(video_frame).astype('uint8'), cv2.COLOR_RGB2BGR))


def project_image(proj, targets, png_prefix, num_snapshots, out_widget=None):
    video_out = cv2.VideoWriter(png_prefix + '_video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 24, (512, 512))
    snapshot_steps = set(proj.num_steps - np.linspace(0, proj.num_steps, num_snapshots, endpoint=False, dtype=int))
    misc.save_image_grid(targets, png_prefix + '_target.png', drange=[-1 ,1])
    proj.start(targets)
    while proj.get_cur_step() < proj.num_steps:
        proj.step()
        imgout = misc.convert_to_pil_image(misc.create_image_grid(proj.get_images(), None), drange=[-1 ,1]).resize((512, 512))
        if proj.get_cur_step() % 10 == 0 and out_widget is not None:
            out_widget.clear_output()
            with out_widget:
                display(imgout)

        write_video_frame(imgout, video_out)
        if proj.get_cur_step() in snapshot_steps:
            misc.save_image_grid(proj.get_images(), png_prefix + '_step%04d.png' % proj.get_cur_step(), drange=[-1 ,1])

    dlats = proj.get_dlatents()
    joblib.dump(dlats, png_prefix + '_dlats.jbl')
    video_out.release()


class WidgetRepo:
    def __init(self):
        self.isinit = True

wrepo = WidgetRepo


def wass_dist_mean(dlat, dlat_vec):
    dists = []
    for dlat1d in dlat:
        dist = wass_dist(dlat_vec, dlat1d)
        dists.append(dist)

    return np.array(dists).mean()


def gather_direc(direcs):
    direc = None
    for sln, sl in WidgetRepo.sliders.items():
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

lids = range(8)
def on_value_change(change):
    try:
        newval = change.new
    except:
        newval = change
    direc = gather_direc()
    dlat = ust.lerp_dir_dlats(WidgetRepo.dlat, direc, [WidgetRepo.slider_coef.value], layer_indices=lids)
    WidgetRepo.dlat_plot.clear_output()
    with WidgetRepo.dlat_plot:
        plot_dlat(dlat[0][lids])

    handle_render({})


def plot_dlat(dlat):
    ax, fig = plt.subplots()
    ax.set_figheight(5)
    ax.set_figwidth(8)
    plt.plot(dlat[lids].T, linestyle='', marker='.', alpha=0.2, markersize=5, antialiased=True)
    # plt.hlines([-2, 2.], xmin=0, xmax=512, colors='k')
    plt.xlim(0, 512)
    plt.ylim(-1.1, 5.1)
    plt.show()


def handle_render(obj):
    direc = gather_direc(wrepo.Gs)
    dlats_n = ust.lerp_dir_dlats(wrepo.dlat, direc, [WidgetRepo.slider_coef.value], layer_indices=lids)
    images = wrepo.Gs.components.synthesis.run(np.array(dlats_n), **wrepo.Gs_syn_kwargs)
    images_pil = [Im.fromarray(img, 'RGB') for img in images]
    WidgetRepo.image_out.clear_output()
    with WidgetRepo.image_out:
        display(images_pil[0].resize((300, 300)))


lids = range(8)


def create_ui():
    WidgetRepo.btn_render = wdgts.Button(description='Render')
    WidgetRepo.btn_render.on_click(handle_render)

    WidgetRepo.sliders = create_sliders(list(wrepo.direcs.keys()))
    WidgetRepo.slider_coef = wdgts.FloatSlider(value=0., min=-15., max=15, step=1, orientation='horizontal',
                                               description='Shift')
    WidgetRepo.slider_coef.observe(on_value_change, names='value')

    WidgetRepo.image_init_out = wdgts.Output()
    WidgetRepo.image_out = wdgts.Output()
    WidgetRepo.dlat_plot = wdgts.Output()

    lay = wdgts.GridspecLayout(1, 2)
    lay[0, 0] = wdgts.VBox([WidgetRepo.slider_coef, WidgetRepo.btn_render] + list(WidgetRepo.sliders.values()))
    lay[0, 1] = wdgts.VBox(
        [wdgts.HBox([WidgetRepo.image_init_out, WidgetRepo.image_out],
                    layout=wdgts.Layout(width='650px', height='350px')), WidgetRepo.dlat_plot],
        layout=wdgts.Layout(width='650px'))

    return lay