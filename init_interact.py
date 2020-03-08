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


wrepo = ui_utils.WidgetRepo()
wrepo.PATH_DATA = PATH_DATA = '/content/drive/My Drive/stylegan2'
wrepo.PATH_RESULTS = PATH_RESULTS = f'{PATH_DATA}/results'
wrepo.PATH_IMG_RAW = PATH_IMG_RAW = f'{PATH_DATA}/img_raw'
wrepo.PATH_IMG = PATH_IMG = f'{PATH_DATA}/img'
PATH_DS = './data/ds'
wrepo.PATH_DIRS = PATH_DIRS = './assets/dirs_dlat'
wrepo.PATH_DLATS = PATH_DLATS = PATH_RESULTS
wrepo.init_Gs()
out_void = Output()