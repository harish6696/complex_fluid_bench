# Complex Fluid Bench
A repository to benchmark two complex flow datasets: Laser Induced Droplet Explosion (LIDE) and Shock Induced Droplet Aerobreakup (SIDA) on different ML models.

# Abstract
Compressible multiphase flow is central to numerous engineering applications, characterized by complex wave dynamics and challenging shock-interface interactions. Despite their importance, they remain significantly missing from existing benchmarks in the Scientific Machine Learning (SciML) community, limiting progress on generalization to impactful real-world scenarios. To address this issue, we introduce two exemplary datasets from this class, Laser-Induced Droplet Explosion (LIDE) and Shock-Induced Droplet Aero-breakup (SIDA), providing researchers with valuable references to establish reliable baselines and push boundaries of SciML. Due to the high computational cost of simulating these processes with full fidelity, we explore data-driven surrogate models designed to efficiently approximate the underlying physics at reduced cost. We benchmark these datasets on diverse architectures–UNet, Fourier Neural Operator (FNO), Vision Transformer (ViT), Scalable Operator Transformer (ScOT), and Residual Network (ResNet)–trained autoregressively and compared across varying parameter counts. A comprehensive set of ablations is carried out to analyze the performance of the models. We identify key scenarios, such as incorporating temporal sequence information and conditioning, that enable the models to accurately capture the rich and nonlinear physics embedded in the datasets.

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

Datasets used in the paper are provided on the 🤗 Hub. You can download them from the following links: 
- [🤗 Hub – LIDE Dataset](https://huggingface.co/datasets/FluidVerse/LIDE)
- [🤗 Hub – SIDA Dataset](https://huggingface.co/datasets/FluidVerse/SIDA)
