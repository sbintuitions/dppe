# [DPPE: Rethinking Camera-Based Positional Encoding for Scaling Multi-View Transformers (NeurIPS2026, Main Track)](https://arxiv.org/abs/2606.31585)

This repository provides the implementation of **Decoupled Pose Positional Encoding (DPPE)**, a positional encoding method for multiview transformers.

DPPE decouples the camera pose information into separate components for query-key and value-output attention streams, enabling more expressive and geometrically-aware attention.

## Setup

### Requirements
Ensure you have `uv` installed (see [official installation guide](https://docs.astral.sh/uv/getting-started/installation/)).

```bash
# Basic setup
uv sync
```
If you wish to use Flash Attention 3, please run:
`uv sync --extra flash-attn-3`.

## Dataset Preparation
This project assumes datasets are in the [World AI format (WAI format)](https://github.com/MapAnything/MapAnything). After downloading each dataset, convert it to WAI format before use.

### MVImgNet2 Preparation
For MVImgNet2, follow these steps:

1. Obtain the URL and PASSWORD by submitting the form at the [MVImgNet2 Repository](https://github.com/GAP-LAB-CUHK-SZ/MVImgNet2.0).

2. Run the following commands:
```bash
cd wai_processing
# Setup uv env for data processing
uv sync

# Download
bash download_scripts/download_mvimgnet2.sh <URL> <PASSWORD>

# Extract
mkdir -p dataset/unzipped/MVImgNet2 && ls dataset/raw/MVImgNet2/*.tar.gz | xargs -I {} -P 4 tar -zxvf {} -C dataset/unzipped/MVImgNet2

# Convert to WAI format
uv run -m scripts.conversion.mvimgnet2
cd ..
```

## Evaluation Data
Prepare the evaluation structure:
```bash
uv run -m scripts.make_MVImgNet2_eval
```

## Training

### Basic Usage

```bash
bash ./scripts/nvs.sh --pe <PE_NAME> --ray_encoding <ENCODING> --gpus <GPU_LIST> --dataset <DATASET>
```

### Available PE Methods

| PE Name | qk_pe | vo_pe |
|---------|-------|-------|
| `RoPE` | `2d` | `none` |
| `CAPE` | `Rt` | `none` |
| `GTA` | `Rt2d` | `Rt2d` |
| `PRoPE` | `p2d` | `p2d` |
| `DPPEdual` | `pIT2d` | `pIT2d` |
| `DPPEtAdd` | `p2d` | `KRtAdd2d` |

### Model Sizes

| Size | `--num_layers` | `--dim_feedforward` |
|------|---------------|---------------------|
| Small | 6 | 1024 |
| Base | 12 | 3072 |
| Large | 24 | 3072 |

```bash
# Small model
bash ./scripts/nvs.sh --pe DPPEdual --dataset MVImgNet2 \
    --num_layers 6 --dim_feedforward 1024

# Base model (default)
bash ./scripts/nvs.sh --pe DPPEdual --dataset MVImgNet2 \

# Large model
bash ./scripts/nvs.sh --pe DPPEdual --dataset MVImgNet2 \
    --num_layers 24 --dim_feedforward 3072
```

<details>
<summary>Specifying qk_pe and vo_pe directly</summary>

Instead of using `--pe`, you can directly specify the positional encoding for query-key and value-output streams:

```bash
bash ./scripts/nvs.sh --qk_pe p2d --vo_pe KRtAdd2d --gpus "0,1" --dataset MVImgNet2
```

This is useful for experimenting with custom PE combinations beyond the named presets.

</details>

<details>
<summary>Dataset options</summary>

- `RealEstate10K`
- `MVImgNet2`
- `spatialvidhq`

</details>

<details>
<summary>Multi-view training</summary>

To train with a variable number of input views (e.g., randomly sampling between 2 and 4 views per iteration):

```bash
bash ./scripts/nvs.sh --pe DPPEdual --dataset MVImgNet2 --mv_train "2 4"
```

</details>

<details>
<summary>Pixel-wise input addition</summary>

The `--ray_encoding` option controls what geometric information is added pixel-wise to the input:

- `plucker`: Plucker ray coordinates
- `raymap`: Raw ray origin + direction map
- `camray`: Camera-relative ray coordinates
- `none`: No pixel-wise addition

```bash
# With Plucker coordinates
bash ./scripts/nvs.sh --pe DPPEdual --ray_encoding plucker --dataset MVImgNet2

# Without pixel-wise addition
bash ./scripts/nvs.sh --pe DPPEdual --ray_encoding none --dataset MVImgNet2
```

</details>

<details>
<summary>Training configuration</summary>

```bash
# Training
bash ./scripts/nvs.sh --pe DPPEdual --dataset MVImgNet2 \

# Resume training from checkpoint
bash ./scripts/nvs.sh --pe DPPEdual --dataset MVImgNet2 \
    --resume path/to/checkpoint.pt
```

</details>

## Testing

### Zoom-in Evaluation

Test the model's ability to render zoomed-in views:

```bash
bash ./scripts/nvs.sh --pe DPPEdual --dataset MVImgNet2 \
    --resume path/to/checkpoint.pt \
    --test-zoom-in '1.5 2 3 5'
```

### Context Views Evaluation

Test with varying numbers of context (input) views:

```bash
bash ./scripts/nvs.sh --pe DPPEdual --dataset MVImgNet2 \
    --resume path/to/checkpoint.pt \
    --test-context-views '4 6 8 10 12'
```

## Citation

```
@misc{kenney2026dpperethinkingcamerabasedpositional,
      title={DPPE: Rethinking Camera-Based Positional Encoding for Scaling Multi-View Transformers}, 
      author={Shun Kenney and Teppei Suzuki},
      year={2026},
      eprint={2606.31585},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.31585}, 
}
```


## Acknowledgements & Licenses

Our implementation is built upon and incorporates code from the following excellent open-source repositories. We are deeply grateful to the authors for making their work available to the community:

* **[PRoPE](https://github.com/liruilong940607/prope/tree/nvs)**: Used as a baseline and framework for our NVS components.
* **[MapAnything](https://github.com/facebookresearch/map-anything/tree/main/data_processing)**: Used for the data processing pipeline and `wai_processing` integration.

Please note that the original code integrated from these external projects remains subject to the terms and conditions of their respective original licenses.
