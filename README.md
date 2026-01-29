
# Denoising Monte Carlo Renders with Diffusion Models

<p align="center">
  <img src="./pics/teaser_jp.jpg" width="100%">
</p>

Physically-based renderings contain Monte-Carlo noise, with variance that increases as the number of rays per pixel decreases. This noise, while zero-mean for good modern renderers, can have heavy tails (most notably, for scenes containing specular or refractive objects). Learned methods for restoring low fidelity renders are highly developed, because suppressing render noise means one can save compute and use fast renders with few rays per pixel. We demonstrate that a diffusion model can denoise low fidelity renders successfully. Furthermore, our method can be conditioned on a variety of natural render information, and this conditioning helps performance. Quantitative experiments show that our method is competitive with SOTA across a range of sampling rates. Qualitative examination of the reconstructions suggests that the image prior applied by a diffusion method strongly favors reconstructions that are like real images -- so have straight shadow boundaries, curved specularities and no fireflies.

Presented at *3DV 2025* [*Link to paper*](https://arxiv.org/abs/2404.00491)

## Building an Environment
- Please consult [*DeepFloyd IF*](https://github.com/deep-floyd/IF), from which this code heavily borrows.
- We were able to build a working environment as follows:
```shell
conda create -n palette python=3.11 -y
conda activate palette
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install tqdm numpy omegaconf matplotlib Pillow huggingface_hub transformers accelerate diffusers tokenizers sentencepiece ftfy beautifulsoup4
pip install git+https://github.com/openai/CLIP.git --no-deps
pip install configargparse piq opencv-python scikit-image noisebase torchmetrics
pip install --upgrade pyfvvdp
pip install --upgrade zarr
```
- If training from scratch, you'll want to download the base diffusion model first:
```shell
from huggingface_hub import snapshot_download
snapshot_download("deepfloyd/IF-II-M-v1.0")
```

## Dataset
- We show an example of loading the [*noisebase*](https://github.com/balintio/noisebase) dataset in this repo (11 channels - noisy, albedo, normals, depth, sample count map). Because our denoiser is single-frame only, we modified the noisebase dataloader to create batches of random frames from any sequence instead of the default video-style loading. Our method should train on just about any renderer's outputs. 


## Training and Inference
- We trained with batch size 12, which fits on a single A40 GPU. 
- **Training command:**

```shell
python run_dn.py  --logdir "logs/" --amp  --doCN --data_path /path/to/noisebase/data --num_workers=6
```
- To run **inference** add the `--eval` flag and set the `--model_load_path` to the path of a checkpoint. Use `--spp` to set the sample count of your test dataset.


## Citation
If you find this work helpful in your research, please consider citing our paper:

```shell
@INPROCEEDINGS {11125569,
author = { Vavilala, Vaibhav and Vasanth, Rahul and Forsyth, David },
booktitle = { 2025 International Conference on 3D Vision (3DV) },
title = {{ Denoising Monte Carlo Renders with Diffusion Models }},
year = {2025},
volume = {},
ISSN = {},
pages = {835-844},
abstract = { Physically-based renderings contain Monte Carlo noise, with variance that increases as the number of rays per pixel decreases. This noise, while zero-mean for good modern renderers, can have heavy tails (most notably, for scenes containing specular or refractive objects). Learned methods for restoring low fidelity renders are highly developed, because suppressing render noise means one can save compute and use fast renders with few rays per pixel. We demonstrate that a diffusion model can denoise low fidelity renders successfully. Furthermore, our method can be conditioned on a variety of natural render information, and this conditioning helps performance. Quantitative experiments show that our method is competitive with SOTA across a range of sampling rates. Qualitative examination of the reconstructions suggests that the image prior applied by a diffusion method strongly favors reconstructions that are “like” real images - so have straight shadow boundaries, curved specularities and no “fireflies.” },
keywords = {Training;Monte Carlo methods;Three-dimensional displays;Noise reduction;Noise;Diffusion models;Rendering (computer graphics);Image restoration;Image reconstruction;Videos},
doi = {10.1109/3DV66043.2025.00082},
url = {https://doi.ieeecomputersociety.org/10.1109/3DV66043.2025.00082},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month =mar}

