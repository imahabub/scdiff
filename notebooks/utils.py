import wandb

TEAM_NAME = 'protein-optimization'
PROJECT_NAME = 'sc_diff'

def get_ckpt_path_from_wandb_artifact(artifact_name):
    run = wandb.init(project=PROJECT_NAME, entity=TEAM_NAME)
    artifact = run.use_artifact(artifact_name)
    artifact_dir = artifact.download()
    ckpt_path = f'{artifact_dir}/model.ckpt'
    return ckpt_path

def get_ckpt_path_from_run_name(run_name):
    artifact_name = f'{run_name}/model:best'  # Fetch the latest version of the model artifact
    return get_ckpt_path_from_wandb_artifact(artifact_name)

# import wandb

# TEAM_NAME = 'protein-optimization'
# PROJECT_NAME = 'sc_diff'

# def get_ckpt_path_from_wandb_artifact(artifact_name, artifact_type='model'):
#     run = wandb.init()
#     artifact = run.use_artifact(artifact_name, type=artifact_type)
#     artifact_dir = artifact.download()
#     ckpt_path = f'{artifact_dir}/model.ckpt'
#     return ckpt_path

# def get_ckpt_path_from_run_name(run_name):
#     artifact_name = f'{TEAM_NAME}/{PROJECT_NAME}/{run_name}'
#     return get_ckpt_path_from_wandb_artifact(artifact_name)