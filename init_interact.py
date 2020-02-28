import dnnlib, pretrained_networks, dnnlib.tflib as tflib, projector_oneiro, projector
import numpy as np, joblib, ipywidgets as wdgts, shutil, sys, os, PIL, PIL.Image as Im, pathlib, copy, matplotlib.pyplot as plt, cv2, argparse, re
import utils_interact as ui_utils,  utils_stylegan as ust, dataset_tool as dstool
from training import dataset, misc
from ipywidgets import GridspecLayout, HBox, VBox, Output
from IPython.display import display
from google.colab import drive, files
from os.path import join as pj
from io import BytesIO
from scipy.stats import wasserstein_distance as wass_dist

PATH_DATA = '/content/drive/My Drive/stylegan2'
PATH_RESULTS = f'{PATH_DATA}/results'
PATH_IMG_RAW = f'{PATH_DATA}/img_raw'
PATH_IMG = f'{PATH_DATA}/img'
PATH_DS = './data/ds'
PATH_DIRS = './assets/dirs_dlat'
#PATH_DLATS = './data/nato'
PATH_DLATS = PATH_RESULTS

wrepo = ui_utils.WidgetRepo()
ui_utils.wrepo = wrepo