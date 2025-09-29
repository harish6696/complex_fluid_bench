# Neptuna
A repository to benchmark two complex flow datasets: Laser Induced Droplet Explosion (LIDE) and Shock Induced Droplet aEROBREAKUP (SIDA) on different ML models.

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