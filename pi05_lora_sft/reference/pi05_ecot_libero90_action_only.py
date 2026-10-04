"""Independent configuration. The upstream pi05_libero config is never changed."""
import dataclasses
from pathlib import Path
from openpi.training import config as oc
from openpi.training import weight_loaders

ROOT = Path(__file__).resolve().parents[1]
NAME = 'pi05_ecot_libero90_action_only'
RUN = 'pi05-ecot-libero90-action-only-seed7-bs32-12k'
ASSET_ID = 'ecot_libero90'

def get_config():
    base = oc.get_config('pi05_libero')
    return dataclasses.replace(
        base, name=NAME, exp_name=RUN, project_name='pi05-ecot-libero90-action-only',
        model=dataclasses.replace(base.model, action_horizon=10),
        data=oc.LeRobotLiberoDataConfig(
            repo_id=ASSET_ID,
            assets=oc.AssetsConfig(asset_id=ASSET_ID),
            base_config=oc.DataConfig(prompt_from_task=False),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(ROOT/'data/pi05_base/params')),
        pytorch_weight_path=None,
        assets_base_dir=str(ROOT/'configs/assets'),
        checkpoint_base_dir=str(ROOT/'checkpoints'),
        batch_size=32, seed=7, num_train_steps=12_000,
        lr_schedule=dataclasses.replace(base.lr_schedule,warmup_steps=1000,peak_lr=2e-5,decay_lr=2e-5),
        save_interval=2_000, keep_period=2_000, log_interval=10,
        ema_decay=0.999,
        policy_metadata={'action_horizon':10, 'action_dim':7, 'ema':True,
                         'dataset':'Embodied-CoT/embodied_features_and_demos_libero',
                         'asset_id':ASSET_ID, 'extra_delta_transform':False,
                         'discrete_state_input':base.model.discrete_state_input},
    )

def register():
    config=get_config()
    oc._CONFIGS_DICT[NAME]=config
    return config
