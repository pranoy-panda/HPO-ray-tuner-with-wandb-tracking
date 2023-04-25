import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from filelock import FileLock
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import ray
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray import air, tune
from ray.air.integrations.wandb import setup_wandb
from ray.air.integrations.wandb import WandbLoggerCallback
import wandb
from glob import glob
import random
import argparse

# local files import
from train_and_eval import *
from utils import *
from model import *
from init import *

# Source: https://docs.ray.io/en/latest/tune/examples/tune-pytorch-cifar.html#tune-pytorch-cifar-ref

'''
tune.Tuner - recommended way of launching hyperparameter tuning jobs

args: trainable: tune.Trainable class object to be tuned
      param_space: search space of tuning job
          One thing to note is that both preprocessor and dataset can be tuned here.
      tune_config: Tuning algorithm specific configs.
          Refer to ray.tune.tune_config.TuneConfig for more info.
      run_config: Runtime configuration that is specific to individual trials.
          If passed, this will overwrite the run config passed to the Trainer,
          if applicable. Refer to ray.air.config.RunConfig for more info.
'''

# trainable class
class WandbTrainable(tune.Trainable):
    def setup(self, config):
        # create group name to indicate the hyp params
        group_name = ''
        for key in config.keys():
            if key!='wandb': group_name+=key+'_'+str(round(config[key],3))+'_'
        # setup
        self.wandb = setup_wandb(
            config, trial_id=self.trial_id, trial_name=self.trial_name, group=group_name,api_key = 'add wandb API Key'
        )

    def step(self):
        return train_cifar(self.config)
    
    def save_checkpoint(self, checkpoint_dir: str):
        pass
    
    def load_checkpoint(self, checkpoint_dir: str):
        pass



def main(device_ids=[0],num_samples=10, max_num_epochs=10, gpus_per_trial=2):

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    # ray.init(_temp_dir="ray_tmp_folder/")
    tuner = tune.Tuner(
        tune.with_resources(
            WandbTrainable,
            resources={"cpu": 2, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig( # defaults: search_alg - random search
            metric="loss", # metric to optimize for (should be reported with 'tune.report()')
            mode="min", # select if the metric has to be minimized or maximized
            scheduler=scheduler, # schedulers improve the overall efficiency of the HPO by terminating unpromising trials early
            num_samples=num_samples, # num_samples: Number of times to sample from the hyperparameter space. Defaults to 1. If `grid_search` is
                                    # provided as an argument, the grid will be repeated `num_samples` of times. If this is -1, (virtually) infinite
                                    # samples are generated until a stopping condition is met.
        ),
        run_config=air.RunConfig(
            name = EXP_NAME,
            local_dir = LOCAL_DIR # we can use cloud storage too (https://docs.ray.io/en/latest/tune/tutorials/tune-storage.html)
        ),
        param_space=CONFIG, # hyp param space to be searched
    )
    results = tuner.fit()

    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    parser = argparse.ArgumentParser(
                    prog='cifar10_hyp_tuning_exps',
                    description='Finds best hyperparameter for classification on CIFAR10 dataset',
                    epilog='Text at the bottom of help')
    parser.add_argument('--device_ids',default="0",type=str)
    parser.add_argument('--num_runs',default=10,type=int)
    parser.add_argument('--max_num_epochs',default=10,type=int)
    parser.add_argument('--gpus_per_trial',default=0,type=int)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids # now torch.cuda.device_count() will reflect the number of GPUS
    device_ids_list = list(map(int,args.device_ids.split(',')))
    main(device_ids_list, args.num_runs, args.max_num_epochs, args.gpus_per_trial)