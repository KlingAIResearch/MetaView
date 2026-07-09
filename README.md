<!-- <div align='center'>
<h1 align="center">[ECCV 2026] MetaView: Monocular Novel View Synthesis with Scale-Aware Implicit Geometry Priors</h1>
    <a href='https://openreview.net/profile?id=%7EYufei_Cai1' target='_blank'>Yufei Cai<sup> 1</sup>&emsp;
    <a href='https://nxsedson.github.io/' target='_blank'>Xuesong Niu<sup> 2*</sup>&emsp;
    <a href='https://openreview.net/profile?id=~Hao_LU8' target='_blank'>Hao Lu<sup> 3</sup>&emsp;
    <a href='https://openreview.net/profile?id=~Kun_Gai1' target='_blank'>Kun Gai<sup> 2</sup>&emsp;
    <a href='https://openreview.net/profile?id=~Kai_Wu7' target='_blank'>Kai Wu<sup> 2†</sup>&emsp;
    <a href='https://guosheng.github.io/' target='_blank'>Guosheng Lin<sup> 1†</sup>&emsp;
</div>


<div align='center'>
    <sup>1 </sup>Nanyang Technological University&emsp;
    <sup>2 </sup>Kuaishou Technology&emsp;

</div>

<div align='center'>
    <sup>3 </sup>The Hong Kong University of Science and Technology (Guangzhou)
</div>

<div align='center'>
    <small><sup>*</sup> Project lead</small>&emsp;
    <small><sup>†</sup> Corresponding author</small>
</div>

<div align="center">
  <p>
    <a href="#" target="_blank"><img src="https://img.shields.io/badge/arXiv-MetaView-red" alt="arXiv link"></a>&nbsp;
    <a href="https://prototypenx.github.io/MetaView/" target="_blank"><img src="https://img.shields.io/badge/Project-Homepage-green" alt="project homepage"></a>&nbsp;
    <a href="#" target="_blank"><img src="https://img.shields.io/badge/🤗 Hugging Face-Models-blue" alt="HF Models"></a>&nbsp;
  </p>
</div> -->

<div align="center">

  <h1 style="margin-bottom: 0.2em;">
    [ECCV 2026] MetaView: Monocular Novel View Synthesis with Scale-Aware Implicit Geometry Priors
  </h1>

  <p style="font-size: 1.1em; margin: 0.2em 0;">
    <a href="https://openreview.net/profile?id=%7EYufei_Cai1" target="_blank">Yufei Cai<sup>1</sup></a>&emsp;
    <a href="https://nxsedson.github.io/" target="_blank">Xuesong Niu<sup>2*</sup></a>&emsp;
    <a href="https://lupaopao.github.io/" target="_blank">Hao Lu<sup>3</sup></a>&emsp;
    <a href="https://openreview.net/profile?id=~Kun_Gai1" target="_blank">Kun Gai<sup>2</sup></a>&emsp;
    <a href="https://openreview.net/profile?id=~Kai_Wu7" target="_blank">Kai Wu<sup>2†</sup></a>&emsp;
    <a href="https://guosheng.github.io/" target="_blank">Guosheng Lin<sup>1†</sup></a>
  </p>

  <p style="font-size: 0.95em; margin: 0.2em 0;">
    <sup>1</sup> Nanyang Technological University &nbsp;·&nbsp;
    <sup>2</sup> Kuaishou Technology &nbsp;·&nbsp;
  </p>
  <p style="font-size: 0.95em; margin: 0.2em 0;">
    <sup>3</sup> The Hong Kong University of Science and Technology (Guangzhou)
  </p>

  <p style="font-size: 0.85em; margin: 0.2em 0;">
    <sup>*</sup> Project lead &nbsp;&nbsp; <sup>†</sup> Corresponding author
  </p>

  <p style="margin: 0.8em 0;">
    <a href="#" target="_blank"><img src="https://img.shields.io/static/v1?label=Arxiv&message=UniVideo&color=red&logo=arxiv" alt="arXiv link"></a>&nbsp;
    <a href="https://prototypenx.github.io/MetaView/" target="_blank"><img src="https://img.shields.io/badge/Project-Homepage-green" alt="project homepage"></a>&nbsp;
    <a href="#" target="_blank"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Models-blue" alt="HF Models"></a>
  </p>

</div>



<!-- Official implementation for the paper "**MetaView**: Monocular Novel View Synthesis with Scale-Aware Implicit Geometry Priors". -->

## Overview

**MetaView** is a diffusion-based framework for high-fidelity monocular novel view synthesis that enables accurate rendering under large view changes from a single image. 

Current generative novel view synthesis methods typically rely on restrictive explicit 3D reconstruction pipelines or use fully implicit scene modeling that suffers from scale drifting and poor geometry consistency. MetaView bridges this gap by combining implicit geometry modeling with minimal yet essential explicit 3D cues:

- **Scale-Aware Implicit Geometry Priors:** We extract hierarchical features and metric depth from a feed-forward geometry perception network (Depth Anything 3). These geometric signals are incorporated into the pretrained MM-DiT backbone (Qwen-Image-Edit) via non-invasive parallel attention layers, regularizing the spatial structure while preserving rich semantic knowledge.
- **Metric Scale Anchoring via Modified RoPE:** To overcome the scale drifting issue prevalent in fully implicit methods, we encode camera parameters into a modified Rotary Positional Encoding (PRoPE) and allocate an extra subspace for the *z*-axis. This explicitly injects metric scale cues, anchoring the generation to a consistent 3D metric space.

Given a single input image and a target camera pose, MetaView synthesizes the corresponding novel view with precise camera controllability, strong geometry consistency, and remarkable cross-domain generalization.
## Environment Setup

### 1. Inference Environment: `metaview`

Create the `metaview` conda environment with dependencies for DiffSynth-Studio and Depth Anything 3:

```bash
git clone https://github.com/KlingAIResearch/MetaView.git
cd MetaView

conda create -n metaview python=3.10 -y
conda activate metaview

pip install -e .

# Additional dependencies for inference
pip install opencv-python pillow openexr_numpy

# Install Depth Anything 3
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd DepthAnything3

pip install -e .
```

> **Note on DiffSynth-Studio Installation:** The `diffsynth/` source code and `pyproject.toml` are included in this package, so `pip install -e .` will install DiffSynth-Studio and its dependencies (transformers, accelerate, safetensors, peft, sentencepiece, etc.) following the [official DiffSynth-Studio installation guide](https://github.com/modelscope/DiffSynth-Studio#installation).
>
> **Note on Depth Anything 3:** Installed from the local `DepthAnything3/` source following the [official DA3 installation guide](https://github.com/ByteDance-Seed/Depth-Anything-3#-installation).

### 2. Evaluation Environment: `metric`

Create the `metric` conda environment with UFM dependencies for running evaluation metrics:

```bash
cd MetaView
git clone --recursive https://github.com/UniFlowMatch/UFM.git
cd UFM

conda create -n metric python=3.10 -y
conda activate metric

# Install UniCeption (UFM submodule dependency)
cd UFM/UniCeption
pip install -e .

# Install UFM
cd ..
pip install -e .

# Additional dependencies for metric evaluation
pip install lpips flow_vis scikit-image scikit-learn matplotlib tqdm
```

## Model Weights

Download the following model weights and place them in `models/`:

| Model | Usage | Download |
|-------|-------|---------|
| **DA3-GIANT-1.1** | 3D feature extraction | [Depth-Anything-3-GIANT](https://huggingface.co/depth-anything/DA3-GIANT-1.1) |
| **DA3NESTED-GIANT-LARGE-1.1** | Dense depth estimation | [Depth-Anything-3-NESTED](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1) |
| **Qwen-Image-Edit** | Base DiT model (auto-downloaded) | [Qwen/Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) |
| **UFM-Refine** | Evaluation dependency (auto-downloaded) | [infinity1096/UFM-Refine](https://huggingface.co/infinity1096/UFM-Refine) |
### Recommended Directory Layout

```
models/
├── Qwen/
│   └── Qwen-Image-Edit/
├── MetaView/
│   └── checkpoint-best.safetensors    # MetaView pretrained checkpoint
└── Depth-Anything-3/
    ├── DA3-GIANT-1.1/
    └── DA3NESTED-GIANT-LARGE-1.1/
```

## Usage

### Inference

Modify the `infer.sh` (or `src/inference.py` directly) to generate a novel view from a single image:

```bash
conda activate metaview

# Edit paths inside before running
bash infer.sh
```

Or run with full CLI arguments:

```bash
python src/inference.py \
    --image_path "./samples/2.png" \
    --ckpt_path "Path/to/MetaView/checkpoint" \
    --da3_giant_path "Path/to/DA3-GIANT/model" \
    --da3_depth_path "Path/to/DA3-NESTED/model" \
    --yaw -30 \
    --pitch 15 \
    --radius 3.5 \ # if None, auto-calculated from center depth
    --output_path "./result/output_novel_view.png"
```

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image_path` | *(required)* | Path to the input image |
| `--ckpt_path` | *(required)* | Path to the pretrained MetaView checkpoint (.safetensors) |
| `--output_path` | `./output_novel_view.png` | Path to save the generated image |
| `--yaw` | `0.0` | Yaw angle in degrees (positive = right, negative = left) |
| `--pitch` | `0.0` | Pitch angle in degrees (positive = up, negative = down) |
| `--radius` | `None` (auto) | Rotation radius. If `None`, auto-calculated from center depth |
| `--da3_giant_path` | *(required)* | Path to DA3 GIANT model |
| `--da3_depth_path` | *(required)* | Path to DA3 depth model |
| `--qwen_path` | `None` (auto) | Base path to Qwen-Image-Edit model |

The output is a side-by-side image: **[Source Image | Generated Novel View]**.

<!-- ### Batch Evaluation

Use `validate_demo.sh` to run batch evaluation on benchmark datasets (e.g., DL3DV-10K, RealEstate10K, Sekai):

```bash
conda activate metaview

# Edit validate_demo.sh to set your dataset paths and GPU IDs
bash validate_demo.sh
``` -->

### Evaluation Metrics

Modify `eval.sh` in the `metric` environment to compute PSNR, SSIM, LPIPS, and proposed DMD:

```bash
conda activate metric

# Edit paths inside before running
bash eval.sh
```

Or run with full CLI arguments:

```bash
python -u src/metric.py \
    --data_path "Path/to/evaluation/folder" \
    > ./eval_output.txt  2>&1 
```

The metric script expects input images to be 3-panel stitched images: **[Source Image | Generated | Ground Truth]**, each panel being 960×528.

**Metrics reported:**
- **PSNR** (Peak Signal Noise Ratio)
- **SSIM** (Structural Similarity)
- **LPIPS** (Learned Perceptual Image Patch Similarity, VGG backbone)
- **DMD** (proposed Dense Matching Distance)

## Citation

```bibtex
@article{metaview,
  title={MetaView: Monocular Novel View Synthesis with Scale-Aware Implicit Geometry Priors},
  author={},
  journal={},
  year={2026}
}
```

## Acknowledgements

This codebase builds upon:
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
- [DepthAnything3](https://github.com/ByteDance-Seed/Depth-Anything-3)
- [UFM](https://github.com/UniFlowMatch/UFM)
