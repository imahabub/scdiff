import wandb
import numpy as np
from cellot.losses.mmd import mmd_distance

def get_ckpt_path_from_artifact_id(artifact_id):
    run = wandb.init()
    artifact = run.use_artifact(artifact_id, type='model')
    artifact_dir = artifact.download()
    ckpt_path = f'{artifact_dir}/model.ckpt'
    return ckpt_path

def compute_mmd_loss(lhs, rhs, gammas):
    return np.mean([mmd_distance(lhs, rhs, g) for g in gammas])

from cellot.data.cell import read_single_anndata
def load_markers(config, n_genes=50):
    data = read_single_anndata(config, path=None)
    key = f'marker_genes-{config.data.condition}-rank'

    # rebuttal preprocessing stored marker genes using
    # a generic marker_genes-condition-rank key
    # instead of e.g. marker_genes-drug-rank
    # let's just patch that here:
    if key not in data.varm:
        key = 'marker_genes-condition-rank'
        print('WARNING: using generic condition marker genes')

    sel_mg = (
        data.varm[key][config.data.target]
        .sort_values()
        .index
    )[:n_genes]
    marker_gene_indices = [i for i, gene in enumerate(data.var_names) if gene in sel_mg]

    return sel_mg, marker_gene_indices