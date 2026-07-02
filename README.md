# Neural Style Transfer

A PyTorch implementation of neural style transfer (Gatys et al., 2015): given a **content image** and a **style image**, it synthesizes a new image that preserves the content's structure while adopting the style's textures, colors, and brushwork. Comes with a CLI ([nst/main.py](nst/main.py)) and a Streamlit web UI ([app.py](app.py)).

| Content | Style | Result |
|---|---|---|
| `images/content.jpg` | `images/style.jpg` | `images/generated_image.jpg` |

## How it works

A pretrained VGG19 convolutional network is used purely as a fixed feature extractor (no training, no gradient updates to its weights). The pixels of the *generated image* itself are the only learnable parameters — we optimize them directly so that the image's VGG features simultaneously match the content image's features and the style image's texture statistics.

### 1. Feature extraction

[nst/model.py](nst/model.py) wraps `torchvision.models.vgg19(pretrained=True).features` and taps it at fixed layers:

- **Content representation**: layer `21` (`conv4_2`). Deep enough to capture object structure and layout while discarding exact pixel values.
- **Style representation**: layers `0, 5, 10, 19, 28` (`conv1_1, conv2_1, conv3_1, conv4_1, conv5_1`). Multiple layers, from shallow to deep, capture texture at multiple scales (fine brushstrokes to large color regions).

### 2. Content loss

For a layer's feature map $F$ of the generated image and $P$ of the content image:

$$\mathcal{L}_{content} = \frac{1}{n}\sum_{i} (F_i - P_i)^2$$

i.e. mean squared error between activations. This pulls the generated image's *structure* toward the content image's structure. Implemented in [`ContentLoss`](nst/losses.py).

### 3. Style loss and the Gram matrix

Style is captured by *correlations between feature channels*, not their spatial arrangement. For a layer with $C$ channels of size $H \times W$, reshape the activations to $F \in \mathbb{R}^{C \times HW}$ and compute the **Gram matrix**:

$$G = \frac{F F^\top}{C \cdot H \cdot W}$$

$G_{jk}$ measures how often channel $j$ and channel $k$ activate together, anywhere in the image — a texture "fingerprint" that's independent of where objects sit. The style loss at one layer is the MSE between the generated image's Gram matrix $G$ and the style image's Gram matrix $A$:

$$\mathcal{L}_{style}^{layer} = \frac{1}{n}\sum_{j,k} (G_{jk} - A_{jk})^2$$

The total style loss sums this over all five style layers. Implemented in [`gram_matrix`](nst/losses.py) and [`StyleLoss`](nst/losses.py).

### 4. Total loss and optimization

$$\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{content} + \beta \cdot \mathcal{L}_{style}$$

where $\alpha$ (`--content-weight`) and $\beta$ (`--style-weight`) trade off structure preservation against style strength. Because $\beta \gg \alpha$ by default ($10^6$ vs $1$), style loss values need to be numerically tiny for the two terms to balance — that's the role of the $1/(CHW)$ normalization in the Gram matrix.

The generated image is initialized as a clone of the content image, then its pixel values are updated directly by backpropagating $\mathcal{L}_{total}$ through the frozen VGG19 network using the **Adam** optimizer (the original paper used L-BFGS; Adam trades a bit of convergence speed for simplicity and easier step-count control). See the training loop in [`main()`](nst/main.py).

## Project structure

```
nst/
  main.py     CLI entry point — parses args, runs the optimization loop, saves the result
  model.py    VGGFeatures — frozen VGG19 feature extractor
  losses.py   ContentLoss, StyleLoss, gram_matrix
  utils.py    load_image — loads, resizes, and tensorizes an image
app.py        Streamlit UI — upload images, tune hyperparameters, watch progress, download result
images/       Example content/style images and default output location
```

## Installation

Requires Python 3.9+.

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is used automatically if available (`torch.cuda.is_available()`); otherwise it falls back to CPU, which is significantly slower.

## Usage

### Web UI

```bash
streamlit run app.py
```

Upload a content and style image in the browser, tune the settings in the sidebar, and click **Run style transfer**. A progress bar tracks optimization steps live, and the result can be downloaded when it finishes.

### CLI

```bash
python nst/main.py --content path/to/content.jpg --style path/to/style.jpg --output path/to/result.jpg
```

| Flag | Default | Meaning |
|---|---|---|
| `--content` | `images/content.jpg` | Path to the content image |
| `--style` | `images/style.jpg` | Path to the style image |
| `--output` | `images/generated_image.jpg` | Where to save the result |
| `--steps` | `300` | Number of optimization iterations |
| `--style-weight` | `1e6` | Weight $\beta$ on the style loss |
| `--content-weight` | `1` | Weight $\alpha$ on the content loss |
| `--lr` | `0.01` | Adam learning rate |
| `--max-size` | `512` | Longest edge (px) images are resized to before optimization |
| `--print-every` | `50` | Print the loss every N steps |

## Benchmarking

[nst/benchmark.py](nst/benchmark.py) times the optimization loop (reusing the same model/loss code as `main.py`) across one or more image sizes, reporting ms/step and peak memory. A few warmup steps are run untimed first to absorb one-off costs like cuDNN autotuning.

```bash
python nst/benchmark.py --max-sizes 256 512 768 --steps 50
```

| Flag | Default | Meaning |
|---|---|---|
| `--content` / `--style` | `images/content.jpg` / `images/style.jpg` | Images to benchmark with |
| `--steps` | `50` | Timed optimization steps |
| `--warmup-steps` | `5` | Untimed steps run before timing starts |
| `--max-sizes` | `256 512` | One or more image sizes to benchmark, e.g. `--max-sizes 256 512 768` |
| `--device` | auto | Force `cpu` or `cuda` |
| `--output` | none | Optional path to dump results as JSON |

Peak memory is read from `torch.cuda.max_memory_allocated` on GPU, or process RSS via `psutil` on CPU.

**Measured on this machine** (CPU only, no CUDA - `torch 2.12.1+cpu`, 30 timed steps + 3 warmup):

| max_size | image dims | ms/step | total (30 steps) | peak memory |
|---|---|---|---|---|
| 256 | 255x192 | 240 ms | 7.2 s | 527 MB |
| 512 | 511x384 | 819 ms | 24.6 s | 1006 MB |

Doubling `--max-size` roughly quadruples pixel count, and cost scales accordingly - ms/step went up ~3.4x and peak memory ~1.9x from 256→512. On a CUDA GPU, expect this to be an order of magnitude faster; re-run the command above on your own hardware to get numbers specific to your setup.

## Tuning tips

- **More style, less structure**: increase `--style-weight` (or decrease `--content-weight`).
- **Blurry or noisy result**: lower `--lr`, or increase `--steps` so the optimizer has more time to converge.
- **Slow / out of memory**: reduce `--max-size` - cost scales roughly quadratically with image dimension.
- Content and style images don't need matching dimensions or aspect ratios; both are independently resized so their longest edge is `--max-size`.
