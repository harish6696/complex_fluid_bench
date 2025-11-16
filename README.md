# Complex Fluid Bench
A repository to benchmark two complex flow datasets: Laser Induced Droplet Explosion (LIDE) and Shock Induced Droplet Aerobreakup (SIDA) on different ML models. 
https://openreview.net/forum?id=GXAsUKNyqN

[![intro-image-(1).png](https://i.postimg.cc/PqgqqqKv/intro-image-(1).png)](https://postimg.cc/qtGrb4Pp)

## Installation
Make sure to comment out lines starting with nvidia* inside environment.yml if there is no nvidia-GPU.

Using miniconda: 
```bash
cd install_dependencies
conda env create -f environment.yml
```
After installation:
```bash
conda activate complex_fluid_bench
```

## Datasets

Datasets used in the paper are provided on the 🤗 Hub: 
- [🤗 Hub – LIDE Dataset](https://huggingface.co/datasets/FluidVerse/LIDE)
- [🤗 Hub – SIDA Dataset](https://huggingface.co/datasets/FluidVerse/SIDA)

## Training

Train uses Hydra. The default run loads configs from `config/defaults.yaml` (FNO-1M on Aerobreakup) and starts training:

```bash
python main.py
```

### CLI overview (Hydra overrides)
- Change model (pick any file under `config/model_config/`):
  - `model_config=UNet/unet_1M` or `model_config=UNet/unet_50M`
  - `model_config=ResNet/resnet_1M` 
  - `model_config=FNO/fno_1M` or `model_config=FNO/fno_50M`
  - `model_config=ScOT/scot_1M` or `model_config=ScOT/scot_50M`
  - `model_config=Transolver/transolver_1M`
  - `model_config=CNO/cno_1M`  or `model_config=CNO/cno_50M`
  - `model_config=ViT/vit_1M` or `model_config=ViT/vit_50M`
- Change dataset config:
  - `data_config=fluids/Aerobreakup/2d_aerobreakup_data_default`
  - `data_config.dataset_`
  - `data_config=fluids/Laser_Droplet/2d_laser_droplet_data_default`
- Common training knobs:
  - `train_config.num_train_epochs=50`
  - `train_config.per_device_train_batch_size=4`
  - `train_config.per_device_eval_batch_size=4`
  - `scheduler_config.lr=1e-3`
  - `train_config.use_cpu=True` (force CPU)
  - `train_config.mix_precision_config.fp16=True` (enable FP16 on GPUs)
- Output/logging:
  - `output_log_config.logging.output_dir=outputs/run_${now:%Y%m%d_%H%M%S}`
  - `output_log_config=wandb_log` to enable W&B, then set:
    - `output_log_config.logging.wandb_project=<your_project>`
    - `output_log_config.logging.wandb_api_key=$WANDB_API_KEY`

### Example runs
- Train UNet-1M on Aerobreakup with custom epochs and output dir:
```bash
python main.py \
  model_config=UNet/unet_1M \
  data_config=fluids/Aerobreakup/2d_aerobreakup_data_default \
  train_config.num_train_epochs=50 \
  output_log_config.logging.output_dir=outputs/UNet_Aerobreakup_${now:%Y%m%d_%H%M%S}
```

- Train FNO-50M on Laser Droplet with smaller batch and different LR:
```bash
python main.py \
  model_config=FNO/fno_50M \
  data_config=fluids/Laser_Droplet/2d_laser_droplet_data_default \
  data_config.dataset_directory_path=/absolute/path/to/your/Laser_Droplet/2D \
  train_config.per_device_train_batch_size=2 \
  scheduler_config.lr=5e-4 \
  output_log_config.logging.output_dir=outputs/FNO50M_Laser_${now:%Y%m%d_%H%M%S}
```

- Enable W&B logging:
```bash
python main.py \
  output_log_config=wandb_log \
  output_log_config.logging.output_dir=outputs/UNet_WB_${now:%Y%m%d_%H%M%S} \
  output_log_config.logging.wandb_project=complex_fluid_bench \
  output_log_config.logging.wandb_api_key=$WANDB_API_KEY
```

- Mixed precision (bf16) :
```bash
python main.py train_config.mix_precision_config.bf16=True
```

### Continuing with inference after training
Set the following to automatically run evaluation rollouts and save plots after training:
```bash
python main.py \
  infer_config.do_infer=True \
  infer_config.n_infer_rollouts=4 \
  infer_config.n_infer_plot_examples=2 \
  output_log_config.logging.output_dir=outputs/TrainThenInfer_${now:%Y%m%d_%H%M%S}
```

## Running inference separately on trained models
Run inference on trained models using a provided YAML configuration file (configs/infer_config/only_inference.yaml).

The config file  specifies the model checkpoint, dataset path, the number of rollouts to perform during inference and the number of examples to plot.

Inference always begins from the initial time step (set infer_from_ic=True and infer_from_random_timesteps=False). Use filter options to select specific trajectories or a range of time steps.

```bash
python only_inference.py --config-name=only_inference.yaml
```

## Licensing
This benchmarking code is released under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0) and is exclusively for non-commercial research and educational purposes.

## Citation
If you use our models, code, or datasets, please consider citing our paper:

```bibtex
@misc{complex_fluid_bench,
      title={Laser- and Shock-induced Droplet Dynamics:
A Machine Learning Benchmark for Complex Multiphase Flows},

