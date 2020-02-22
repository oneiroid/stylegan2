import dnnlib, pretrained_networks, dnnlib.tflib as tflib, utils_stylegan as ust, projector_oneiro, projector, numpy as np, joblib, ipywidgets as wdgts
import shutil, dataset_tool as dstool, sys, PIL, PIL.Image as Im, pathlib
from training import dataset, misc
from ipywidgets import GridspecLayout, HBox, Output
from IPython.display import display
from google.colab import drive, files
#import utils_interact as ui_utils

PATH_DATA = '/content/drive/My Drive/stylegan2'
PATH_RESULTS = f'{PATH_DATA}/results'
PATH_IMG_RAW = f'{PATH_DATA}/img_raw'
PATH_IMG = f'{PATH_DATA}/img'
PATH_DS = './data/ds'