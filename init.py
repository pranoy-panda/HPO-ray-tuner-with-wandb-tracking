import torch
import random
import numpy as np
from ray import tune

# for reproducibility
torch.backends.cudnn.deterministic = True
random.seed(2022)
np.random.seed(2022)
torch.manual_seed(2022)
torch.cuda.manual_seed_all(2022)

# paths
LOCAL_DIR = "ray_results/"
EXP_NAME = "experiment_name"

# hyp param tuning search space
CONFIG = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16]),
        "wandb": {"project": "cifar10_exps_multi_gpu"},
        "epochs":10
    }