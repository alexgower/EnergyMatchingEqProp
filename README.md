## Running the examples
- A simple 2D playground is provided in `experiments/toy2d/tutorial_2D.ipynb`.
- A checkpoint for MNIST is provided under `experiments/mnist/assets/`. The CIFAR‑10 checkpoint is too large to include but can be trained using the commands below.
- The CIFAR‑10 checkpoint is too large to include but can be trained using the commands below.

### CIFAR‑10 training and evaluation
Initial training:
@@ -43,28 +43,6 @@ torchrun --nproc_per_node=2 experiments/cifar10/fid_cifar_heun_multigpu.py \
The dataset path defaults to `./data` or can be overridden with the
`CIFAR10_PATH` environment variable.
No CIFAR‑10 checkpoint is included because the file is large, so training from scratch is required or you may provide your own checkpoint.

### ImageNet32 training
Train an ImageNet32 model with:
```bash
torchrun --nproc_per_node=8 experiments/imagenet32/train_imagenet_multigpu.py \
    --lr 6e-4 --batch_size 128
```
The dataset path defaults to `./data/imagenet32` or can be overridden with the
`IMAGENET32_PATH` environment variable.

### ImageNet32 FID evaluation
Evaluate the model using the multi-GPU FID script:
```bash
torchrun --nproc_per_node=2 experiments/imagenet32/fid_imagenet_heun_multigpu.py \
    --resume_ckpt='/path/to/checkpoint.pt' \
    --output_dir=./sampling_results \
    --use_ema True \
    --time_cutoff 0.9 \
    --epsilon_max 0.01 \
    --batch_size 128 \
    --dt_gibbs 0.005
```
### Protein inverse design
Train the model and sample sequences with:
```bash
@@ -73,23 +51,7 @@ python experiments/proteins/sampling_latent.py
```
The VAE used for the continuous latent space is already provided.

### LID on MNIST
Evaluate the local intrinsic dimension on MNIST by running:
```bash
python experiments/mnist/lid_mnist.py
```

### Train the MNIST model
You can also train the MNIST energy model yourself. A minimal command looks like:
```bash
python experiments/mnist/train_mnist.py --lr 1e-4 --batch_size 128
```

### LID on CIFAR‑10
Run the local intrinsic dimension experiment with:
```bash
python experiments/cifar10/lid_cifar.py --chunk_size 64 --resume_ckpt=/path/to/checkpoint.pt --output_dir results_lid_merged --num_samples_test 64 --num_samples_select 64 "$@"
```

## Citation
