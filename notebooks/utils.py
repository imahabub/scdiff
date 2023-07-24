import wandb

def get_ckpt_path_from_wandb_artifact(artifact_name, artifact_type='model'):
    run = wandb.init()
    artifact = run.use_artifact(artifact_name, type=artifact_type)
    artifact_dir = artifact.download()
    ckpt_path = f'{artifact_dir}/model.ckpt'
    return  ckpt_path