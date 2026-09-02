# Energy Matching 
<img align="right" src="media/EM_2D.png" width="30%" alt="Energy Matching Illustration" />
Energy Matching unifies flow matching and energy-based models in a single time-independent scalar field, enabling efficient transport between the source and target distributions while retaining explicit likelihood information for flexible, high-quality generation. [NeurIPS 2025]

**Version 0.9** – This is the official repository for the paper [Energy Matching](https://arxiv.org/abs/2504.10612).

> **This fork** adds backprop-free training and generation for the same models,
> by running the Energy Matching potential as a predictive-coding network. The
> upstream sections below are unchanged; see
> [Backprop-free training and inference](#backprop-free-training-and-inference-this-fork)
> for the added commands.

### Checkpoints
- **CIFAR-10** (Image → Scalar, 50M parameters): warm-up and main-training checkpoints on [Hugging Face](https://huggingface.co/m1balcerak/energy_matching) reach **FID ≈ 3.3** around `T=3.25`.
- **ImageNet32** (Image → Scalar, 50M parameters): warm-up and main-training checkpoints on [Hugging Face](https://huggingface.co/m1balcerak/energy_matching) reach **FID ≈ 6.6** around `T=2.50`.

### Setup (CUDA)
1. Create and activate a Python environment (conda example):
   ```bash
   conda create -n energy-matching python=3.10 -y
   conda activate energy-matching
   ```
2. Install PyTorch with CUDA support and the project requirements:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements.txt
   ```

## Running the examples

### 2D Playground – Eight Gaussians to Two Moons
Experiment with the core idea in a lightweight setting using the notebook at `experiments/toy2d/tutorial_2D.ipynb`. It visualizes how the potential energy field transports particles in 2D from eight Gaussians to two moons.

### CIFAR‑10 Training and Evaluation

<p align="center">
  <strong>Langevin MCMC (unconditional) <br> Trajectory from T = 0 to T = 4 (FID = 3.3)</strong><br>
  <img src="media/cifar10_FID_3_3.gif" width="60%" alt="Trajectory Animation">
</p>

Initial training (warm-up, Algorithm 1):
```bash
torchrun --nproc_per_node=4 experiments/cifar10/train_cifar_multigpu.py \
    --total_steps 145000 \
    --lr 1.2e-3 \
    --batch_size 128 \
    --time_cutoff 1.0 \
    --epsilon_max 0.0 \
    --lambda_cd 0. \
    --n_gibbs 0 \
    --ema_decay 0.9999 \
    --save_step 5000
```
Main training with contrastive divergance (Algorithm 2):
```bash
torchrun --nproc_per_node=4 experiments/cifar10/train_cifar_multigpu.py \
    --resume_ckpt /PATH/TO/warm_up_checkpoint.pt \
    --total_steps 147000 \
    --lr 1.2e-3 \
    --batch_size 128 \
    --time_cutoff 1.0 \
    --epsilon_max 0.01 \
    --lambda_cd 1e-3 \
    --n_gibbs 200 \
    --ema_decay 0.99 \
    --save_step 100 \
    --dt_gibbs 0.01 \
    --cd_neg_clamp 0.02  \
    --split_negative True \
    --same_temperature_scheduler True
```
Evaluation FID across trajectories at times `T=1.0` to `T=5.0` (Heun solver):
```bash
python experiments/cifar10/fid_cifar_heun_1gpu.py \
    --resume_ckpt=/PATH/TO/main_training_checkpoint.pt \
    --output_dir=./sampling_results \
    --use_ema True \
    --time_cutoff 1.0 \
    --epsilon_max 0.01 \
    --batch_size 64 \
    --dt_gibbs 0.01
```
Pretrained CIFAR-10 checkpoints are available at [Hugging Face](https://huggingface.co/m1balcerak/energy_matching_cifar10).
Use `cifar10_warm_up_145000.pt` for the warm-up phase and `cifar10_main_training_147000.pt` after the main training. The latter obtains an **FID of 3.3** at around `T=3.25`.

To generate CIFAR-10 images using unconditional Langevin Monte Carlo sampling from the trained Energy Matching model, run:

```bash
python experiments/cifar10/sample_cifar_heun_1gpu.py \
    --resume_ckpt=/PATH/TO/main_training_checkpoint.pt \
    --batch_size 128 \
    --time_cutoff 1.0 \
    --epsilon_max 0.01 \
    --dt_gibbs 0.01 \
    --use_ema True \
    --t_end=3.25
```
Here, `t_end` corresponds to the sampling time $\tau_s$.

### ImageNet32 Training and Evaluation

Download the downsampled ImageNet32 training batches (`train_data_batch_1` ... `train_data_batch_10`) from the [official release](https://patrykchrabaszcz.github.io/Imagenet32/) and place them under `experiments/imagenet/data/Imagenet32_train/` (or point the `IMAGENET32_PATH` environment variable to that folder) before launching training.

Initial training (Algorithm 1):
```bash
torchrun --nproc_per_node=7 experiments/imagenet/train_imagenet_multigpu.py \
    --total_steps 640000 \
    --lr 6e-4 \
    --batch_size 128 \
    --time_cutoff 1.0 \
    --epsilon_max 0. \
    --lambda_cd 0. \
    --ema_decay 0.9999 \
    --save_step 80000
```
Main training with contrastive divergence (Algorithm 2):
```bash
torchrun --nproc_per_node=7 experiments/imagenet/train_imagenet_multigpu.py \
    --resume_ckpt=/PATH/TO/warm_up_imagenet_checkpoint.pt \
    --total_steps 641000 \
    --lr 6e-4 \
    --batch_size 128 \
    --time_cutoff 1.0 \
    --epsilon_max 0.01 \
    --lambda_cd 0.001 \
    --n_gibbs 200 \
    --ema_decay 0.99 \
    --save_step 100 \
    --cd_neg_clamp 0.02 \
    --split_negative True \
    --same_temperature_scheduler True
```
FID evaluation across trajectory times `T=0.75` to `T=4.0` (Heun solver):
```bash
torchrun --nproc_per_node=1 experiments/imagenet/fid_imagenet_heun_multigpu.py \
    --resume_ckpt=/PATH/TO/main_training_imagenet_checkpoint.pt \
    --output_dir=./sampling_results \
    --use_ema True \
    --time_cutoff 1.0 \
    --epsilon_max 0.01 \
    --batch_size 128 \
    --dt_gibbs 0.01
```
Pretrained ImageNet32 checkpoints (warm-up and main training) are hosted on [Hugging Face](https://huggingface.co/m1balcerak/energy_matching).


### Protein inverse design
Train the model with:
```bash
python experiments/proteins/train_proteins.py \
    --epsilon_max 0.1 \
    --time_cutoff 0.9 \
    --n_gibbs 200 \
    --dt_gibbs 0.01
```
Pretrained AAV medium/hard checkpoints are available at [Hugging Face](https://huggingface.co/m1balcerak/energy_matching). Run conditional sampling with:
```bash
python experiments/proteins/sampling.py
```
The VAE used for the continuous latent space and the dataset is already provided. 



## Backprop-free training and inference (this fork)

This fork extends Energy Matching so that both the **generation** loop and the
**training** loop can run without backpropagation, by re-expressing the same
50M-parameter potential as a predictive-coding network (PCN) whose relaxation
fixed point reproduces the feedforward velocity.

Two independent claims, in increasing order of strength:

1. **Backprop-free generation.** Energy Matching sampling needs `grad_x V` at
   every ODE step. Loading an *unmodified* Energy Matching checkpoint into the
   PCN and sampling by relaxation alone gives the same FID.
2. **Backprop-free training.** The same network trained end to end by
   equilibrium propagation (EP) or by an implicit-function (IFT) rule, with no
   backward pass through the network at any point.

Everything below is CIFAR-10 unless stated. Commands are given without a job
scheduler, matching the style of the sections above; we run them under `uv`, so
prefix with `uv run` or activate an equivalent environment first.

### Conventions (read before quoting any number)

- **Clamp strength.** The paper states the clamp on the potential `V` with
  strength `gamma_V`; the code parametrises it on the pre-scale scalar
  `o = V/alpha`, so `gamma_code = alpha * gamma_V`, with `alpha = output_scale`
  = 1000 (CIFAR-10, ImageNet-32) and 100 (MNIST). Every command below uses code
  units. The recipe point `--pcn_gamma=0.1` is `gamma_V = 1e-4`.
- **FID protocols.** 50k samples is the only quotable "report" protocol; the
  5k/10k variants are ranking instruments only. Never mix them in one table.
  `--fid_seed` fixes the sampling noise so two samplers can be compared as a
  matched pair.
- **TF32** matmul kernels are enabled throughout, which sets a relative velocity
  error floor near 2e-4. This is far above the algorithmic bias at the recipe
  `gamma`, and is why the correspondence sweeps report a float64 pass too.

### 1. Replication of the Energy Matching baseline

Warm-up (Algorithm 1) at effective batch 1024 = 4 GPUs x 128 x accum 2:

```bash
torchrun --standalone --nproc_per_node=4 experiments/cifar10/train_cifar_multigpu.py \
    --grad_accum=2 \
    --total_steps=145000 --batch_size=128 --lr=1.2e-3 \
    --warmup=10000 --ema_decay=0.9999 --save_step=5000 \
    --output_dir=./results_cifar10_ffn/main/replication
```

Phase 2, contrastive divergence (Algorithm 2), effective batch 512, no accum:

```bash
torchrun --standalone --nproc_per_node=4 experiments/cifar10/train_cifar_multigpu.py \
    --resume_ckpt=/PATH/TO/checkpoint_145000.pt \
    --total_steps=147000 --batch_size=128 --lr=1.2e-3 --warmup=10000 \
    --ema_decay=0.99 \
    --lambda_cd=1e-3 --n_gibbs=200 --dt_gibbs=0.01 --epsilon_max=0.01 \
    --time_cutoff=1.0 --cd_trim_fraction=0.1 --cd_neg_clamp=0.02 \
    --split_negative --same_temperature_scheduler \
    --save_step=250 --output_dir=./results_cifar10_ffn/main/replication_p2cd
```

Report-protocol FID with the ordinary autograd sampler (`--fid_times=3.25` for
the post-CD checkpoint):

```bash
torchrun --standalone --nproc_per_node=4 experiments/cifar10/fid_cifar_heun_multigpu.py \
    --resume_ckpt=/PATH/TO/checkpoint.pt --use_ema \
    --n_samples=50000 --fid_times=1.0 --fid_seed=1 \
    --batch_size=128 --num_workers=4 --dt_gibbs=0.01 --epsilon_max=0.01 --time_cutoff=1.0
```

### 2. Backprop-free generation: train with backprop, sample with the PCN

The same checkpoint, remapped into the PCN by `--ffn_checkpoint_into_pcn` and
sampled by relaxation. Note the FID sample-count flag is `--fid_n_samples` here,
unlike the feedforward script above.

```bash
torchrun --standalone --nproc_per_node=4 experiments/cifar10_pcn/fid_cifar_heun_multigpu.py \
    --model_type=pcn_unet_vit --pcn_error_param --param_grad_mode=ift \
    --ffn_checkpoint_into_pcn \
    --pcn_gamma=0.1 --K_h=1 --T_free=14 --pcn_cg_steps=3 --pcn_dt=1.0 \
    --resume_ckpt=/PATH/TO/checkpoint.pt --use_ema \
    --fid_n_samples=50000 --fid_times=1.0 --fid_seed=1 \
    --batch_size=128 --num_workers=4 --dt_gibbs=0.01 --epsilon_max=0.01 --time_cutoff=1.0
```

Use `--fid_times=3.25` for the post-CD checkpoint. Sampling by relaxation is
roughly an order of magnitude slower than the autograd sampler.

### 3. Backprop-free training

Implicit-function (IFT) rule, 4 GPUs, effective batch 1024:

```bash
torchrun --standalone --nproc_per_node=4 experiments/cifar10_pcn/train_cifar_multigpu.py \
    --model_type=pcn_unet_vit --pcn_error_param --param_grad_mode=ift \
    --pcn_frozen_dropout=0.1 --pcn_gamma=0.1 \
    --K_h=2 --T_free=15 --pcn_cg_steps=10 --pcn_dt=1.0 \
    --grad_skip_threshold=100 \
    --batch_size=128 --grad_accum=2 \
    --total_steps=145000 --lr=1.2e-3 --warmup=10000 \
    --ema_decay=0.9999 --gen_ema --save_step=1000 \
    --output_dir=./results_cifar10_pcn/main/ift_main
```

Equilibrium propagation (EP) with the linearised nudge, across 2 nodes x 4 GPUs
for the same effective batch of 1024 without accumulation:

```bash
torchrun --nnodes=2 --nproc_per_node=4 \
    --rdzv_backend=c10d --rdzv_endpoint=$HEAD_NODE:29513 \
    experiments/cifar10_pcn/train_cifar_multigpu.py \
    --model_type=pcn_unet_vit --pcn_error_param --param_grad_mode=ep \
    --nudge_type=linear --pcn_frozen_dropout=0.1 \
    --pcn_gamma=0.1 --lambda_spring=1.0 --beta=30 \
    --K_h=1 --T_free=14 --T_nudge=6 --pcn_dt=1.0 --thirdphase \
    --grad_skip_threshold=100 \
    --batch_size=128 --grad_accum=1 \
    --total_steps=145000 --lr=1.2e-3 --warmup=10000 \
    --ema_decay=0.9999 --gen_ema --save_step=1000 \
    --output_dir=./results_cifar10_pcn/main/ep_main
```

Phase 2 for a PCN-trained arm. Phase 2 holds three live relaxation graphs (flow,
positive and negative energy), so the per-GPU batch drops to 64 with accum 2 to
stay inside 80 GB; run with `PYTORCH_ALLOC_CONF=expandable_segments:True`:

```bash
torchrun --standalone --nproc_per_node=4 experiments/cifar10_pcn/train_cifar_multigpu.py \
    --model_type=pcn_unet_vit --pcn_error_param --param_grad_mode=ift \
    --pcn_frozen_dropout=0.1 --pcn_gamma=0.1 \
    --K_h=2 --T_free=15 --pcn_cg_steps=10 --pcn_dt=1.0 \
    --grad_skip_threshold=100 \
    --resume_ckpt=/PATH/TO/checkpoint_145000.pt \
    --total_steps=147000 --batch_size=64 --grad_accum=2 --lr=1.2e-3 --warmup=10000 \
    --ema_decay=0.99 \
    --lambda_cd=1e-3 --n_gibbs=200 --dt_gibbs=0.01 --epsilon_max=0.01 \
    --time_cutoff=1.0 --cd_trim_fraction=0.1 --cd_neg_clamp=0.02 \
    --split_negative --same_temperature_scheduler \
    --save_step=250 --output_dir=./results_cifar10_pcn/main/ift_main_p2cd
```

### 4. Supporting experiments and paper figures

Every other claim in the paper has its own directory under `results_*/in_paper/`
holding the raw sweep logs, the figures, and an `EXPLANATION.md` that records the
exact command, the checkpoint used, and the provenance of each quoted number,
including superseded values and why they were retired. Where a claim has figures,
its directory also carries a standalone `make_figures.py` that regenerates them
from the committed logs using only numpy and matplotlib.

These currently cover the velocity correspondence as a function of clamp
strength, relaxation budget and arithmetic precision; the matched-seed FID
comparison behind the porting claim above; FID against training step for both
backprop-free arms; and a node-granularity sweep on MNIST. Start from the
`EXPLANATION.md` in the relevant directory.


## Citation

If you find our work useful, please consider citing:

```bibtex
@article{balcerak2025energy,
  title={Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling},
  author={Balcerak, Michal and Amiranashvili, Tamaz and Terpin, Antonio and Shit, Suprosanna and Bogensperger, Lea and Kaltenbach, Sebastian and Koumoutsakos, Petros and Menze, Bjoern},
  journal={arXiv preprint arXiv:2504.10612},
  year={2025}
}
```
