'''
Purpose of this script is to execute different ways of integrating wandb and ray
Source: https://docs.ray.io/en/latest/tune/examples/tune-wandb.html
'''

# step 0: wandb login

import numpy as np

import ray
from ray import air, tune
from ray.air import session
from ray.air.integrations.wandb import setup_wandb
from ray.air.integrations.wandb import WandbLoggerCallback
import os
import torch
import random

# for reproducibility
torch.backends.cudnn.deterministic = True
random.seed(2022)
np.random.seed(2022)
torch.manual_seed(2022)
torch.cuda.manual_seed_all(2022)

# Method 1: wandb integration via the callback function
def train_function(config):
    '''
    reports loss to Tune
    '''
    for i in range(30):
        loss = config["mean"] + config["sd"] * np.random.randn()
        session.report({"loss": loss})


def tune_with_callback():
    """Example for using a WandbLoggerCallback with the function API"""
    tuner = tune.Tuner(
        train_function,
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
        ),
        run_config=air.RunConfig(
            callbacks=[
                WandbLoggerCallback(project="Wandb_example")
            ]
        ),
        param_space={
            "mean": tune.grid_search([1, 2, 3, 4, 5]),
            "sd": tune.uniform(0.2, 0.8),
        },
    )
    tuner.fit() # start the hyp param tuning job


# Method 2: wandb logging in the train_func itself
def train_function_wandb(config):
    wandb = setup_wandb(config)

    for i in range(30):
        loss = config["mean"] + config["sd"] * np.random.randn()
        session.report({"loss": loss})
        wandb.log(dict(loss=loss))
    return {"loss": loss, "done": True}


def tune_with_setup():
    """Example for using the setup_wandb utility with the function API"""
    tuner = tune.Tuner(
        train_function_wandb,
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
        ),
        param_space={
            "mean": tune.grid_search([1, 2, 3, 4, 5]),
            "sd": tune.uniform(0.2, 0.8),
            "wandb": {"project": "integration_with_ray"},
        },
    )
    tuner.fit()


# Method 3: class
class WandbTrainable(tune.Trainable):
    def setup(self, config):
        self.wandb = setup_wandb(
            config, trial_id=self.trial_id, trial_name=self.trial_name, group=str(self.trial_id)
        )

    def step(self):
        return train_function_wandb(self.config)
    
    def save_checkpoint(self, checkpoint_dir: str):
        pass
    
    def load_checkpoint(self, checkpoint_dir: str):
        pass


def tune_trainable():
    """Example for using a WandTrainableMixin with the class API"""
    tuner = tune.Tuner(
        WandbTrainable,
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
        ),
        param_space={
            "mean": tune.grid_search([1, 2, 3, 4, 5,6]),
            "sd": tune.uniform(0.2, 0.8),
            "wandb": {"project": "integration_with_ray"},
        },
    )

    results = tuner.fit()

    return results.get_best_result().config


mock_api = False

if mock_api:
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_API_KEY", "abcd")
    ray.init(
        runtime_env={
            "env_vars": {"WANDB_MODE": "disabled", "WANDB_API_KEY": "abcd"}
        }
    )

# tune_with_callback()
# tune_with_setup()
tune_trainable()


