import streamlit as st
import argparse
from default_config import get_default_config
from scripts.default_config import (
    model_kwargs,
    imagedata_kwargs
)
import torchreid
from torchreid.utils import (
    load_pretrained_weights,
    check_isfile
)
from torchreid import metrics
import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os.path as Path
import numpy as np


parser = argparse.ArgumentParser(
    description="ReID Demo"
)
parser.add_argument(
    "-r", "--root",
    type=str,
    required=True,
    help="Path to data..."
)


transforms = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@st.cache
def feature_extraction(model, data_loader):
    _features, _pids = [], []
    for b_idx, data in enumerate(data_loader):
        imgs, pids = data
        
        imgs = imgs.cuda()
        outputs = model(imgs).data.cpu()

        _features.append(outputs)
        _pids.append(pids)
    _features = torch.cat(_features, 0)
    _pids = np.asarray(_pids)

    return _features, _pids


@st.cache
def compute_distance_matrix(query_features, gallery_features, metric="cosine"):
    return metrics.compute_distance_matrix(query_features, gallery_features).numpy()


@st.cache
def visualize(distmat, query_set, gallery_set):
    pass


def main(args):
    st.header("Person ReID Demo")
    st.sidebar.title("Model information:")
    config_path = st.sidebar.text_input("Path to NN config")
    weights = st.sidebar.text_input("Path to NN weights")

    cfg = get_default_config()
    if config_path:
        cfg.merge_from_file(config_path)
    model = torchreid.models.build_model(**model_kwargs(cfg, 0))

    if weights and check_isfile(weights):
        load_pretrained_weights(model, weights)

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True
        model = model.cuda()

    query_dataset = ImageFolder(
        Path.join(args.root, "query"),
        transforms
    )
    gallery_dataset = ImageFolder(
        Path.join(args.root, "gallery"),
        transforms
    )

    num_pids = len(query_dataset)

    query_loader = DataLoader(query_dataset)
    gallery_loader = DataLoader(gallery_dataset)
    
    qf, qp = feature_extraction(model, query_loader)
    gf, gp = feature_extraction(model, gallery_loader)

    distmat = compute_distance_matrix(qf, gf)

    indices = np.argsort(distmat, axis=1)

    idx = st.slider(
        "Image",
        min_value=0,
        max_value=num_pids - 1,
        step=1
    )
    # need to use slider

