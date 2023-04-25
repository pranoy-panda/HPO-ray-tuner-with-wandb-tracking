## Ray Tune with wandb
[https://docs.ray.io/en/latest/tune/examples/tune-wandb.html]

Two integrations related functions: 
- ```WandbLoggerCallback``` - automatically logs metrics reported to Tune to the Wandb API.
- ```setup_wandb()``` - initializes the Wandb API with Tune’s training information.

There are multiple ways of integrating wandb with ray (see [here](test_ray_tune_with_wandb_logging.py)), however I prefer the [class-based method](test_ray_tune_with_wandb_logging.py#84) as it is capable of handling more functionalities and ```tune.run``` will be depricated soon.

## Commandline
python test_cifar10_ray_with_wandb.py --device_ids 0 --num_runs 5 --max_num_epochs 10 --gpus_per_trial 0

## Ray tune

Important functions:
- ray.tuner

errors stored in /root/ray_results

sanity checks before using ray tuner:
- for all the pts in the hyp param search space the code should run without error
- overfit the designed model for some hyp params to check if the code is correct or not

## Wandb


