````markdown
# NeuroCLIP

NeuroCLIP is a multimodal learning project that aligns scientific language with 3D neural activation patterns. The project explores how contrastive vision-language learning can support natural language-based retrieval and interpretation of neural imaging data.

This repository contains the research prototype developed for experiments on cross-modal representation learning between textual descriptions and 3D brain activation maps.

## Overview

Modern neuroscience and biomedical datasets often contain both textual descriptions, such as article abstracts or scientific annotations, and 3D neural activation patterns. However, these modalities are often analyzed separately. NeuroCLIP aims to learn a shared embedding space where scientific text and 3D neural data can be compared directly.

The model is inspired by CLIP-style contrastive learning, but adapted to 3D neural imaging data. It combines:

- a 3D visual encoder for neural activation maps;
- a Transformer-based text encoder for scientific language;
- global contrastive image-text alignment;
- local image-text alignment for finer-grained multimodal representation learning.

## Motivation

The main goal of this project is to investigate whether scientific language can be aligned with neural activation patterns in a way that enables more intuitive exploration of neuroscience data.

Potential use cases include:

- retrieving relevant neural activation maps from natural language queries;
- retrieving scientific descriptions from neural activation patterns;
- supporting interpretable exploration of brain imaging datasets;
- studying cross-modal learning between biomedical text and 3D neural data.

## Model Architecture

NeuroCLIP consists of three main components.

### 1. 3D Image Encoder

The visual encoder processes 3D neural activation maps using a 3D convolutional architecture. The current implementation includes a ResNet3D-style encoder with attention pooling to produce global image features, while preserving local visual features for local alignment.

An experimental 3D Vision Transformer encoder is also included for future comparison and extension.

### 2. Text Encoder

The text encoder is a Transformer-based model for scientific language representation. It uses modern architectural components such as:

- RMSNorm;
- rotary positional embeddings;
- SwiGLU-style feed-forward layers;
- causal self-attention;
- scaled dot-product attention when available.

The encoder produces both global text embeddings and local token-level representations.

### 3. Global and Local Alignment

The model uses two complementary learning objectives:

- **Global alignment:** a CLIP-style contrastive objective aligns global image and text representations.
- **Local alignment:** local visual features are combined with token-level text features to support finer-grained multimodal representation learning.

This design allows the model to learn both high-level image-text similarity and more detailed cross-modal interactions.

## Repository Structure

```text
.
├── NeuronClip.py          # Main NeuroCLIP model and alignment objectives
├── Resnet_encoder.py      # 3D ResNet-style visual encoder
├── ViT_3D.py              # Experimental 3D Vision Transformer encoder
├── text_encoder.py        # Transformer-based text encoder for global alignment
├── text_generation.py     # Transformer module for local text-conditioned modeling
├── dataloader.py          # Dataset loading and preprocessing utilities
├── trainer.py             # Training loop, evaluation, and checkpointing utilities
├── main.py                # Hydra-based training entry point
└── README.md
````

## Key Features

* CLIP-style contrastive learning for 3D neural data and scientific text.
* 3D image encoding with residual convolutional blocks and attention pooling.
* Transformer-based text encoding with RMSNorm and rotary embeddings.
* Local image-text alignment for multimodal representation learning.
* Retrieval evaluation for image-to-text and text-to-image matching.
* Experimental support for both 3D ResNet-style and 3D Vision Transformer encoders.


## Installation

Clone the repository:

```bash
git clone https://github.com/trandat27jk/NeuroClip.git
cd NeuroClip
```

Install the main dependencies:

```bash
pip install torch torchvision torchaudio
pip install transformers tiktoken einops rotary-embedding-torch hydra-core omegaconf pandas numpy fsspec
```

Alternatively, create a `requirements.txt` file with the following dependencies:

```text
torch
torchvision
torchaudio
transformers
tiktoken
einops
rotary-embedding-torch
hydra-core
omegaconf
pandas
numpy
fsspec
```

## Data Preparation

The project expects paired scientific text and 3D neural activation maps. In the original experimental setup, 3D activation maps are stored as `.npy` files and paired with textual descriptions using article or PMID identifiers.

Expected directory structure:

```text
data/
└── convert_img/
    ├── pmid_XXXX.npy
    ├── pmid_YYYY.npy
    └── ...
```

The text metadata is expected in JSON/CSV format, for example:

```text
neuronClip_data.json
processed_data.csv
testing_data.csv
```

Each `.npy` file represents a 3D neural activation map, while the JSON/CSV metadata files provide the corresponding textual descriptions or article-level information.

## Data Availability and Model Weights

The dataset used in this project is internal and cannot be publicly released due to data access and licensing constraints. As a result, this repository does not include the full training data, preprocessing files, or raw 3D neural activation maps.

For academic or research-related inquiries, model weights and additional experimental details may be shared upon reasonable request. Please contact:

**Van Dat Tran**
Email: [dat.tran220407@vnuk.edu.vn](mailto:dat.tran220407@vnuk.edu.vn)

## Training

The training pipeline is being cleaned for easier reproducibility. In the original experimental setup, training was launched through a Hydra-based entry point after preparing the dataset paths and model configuration:

```bash
python main.py
```

The model is trained with two complementary objectives:

* global CLIP-style contrastive image-text alignment;
* local image-text alignment for text-conditioned multimodal representation learning.

The trainer includes support for:

* mixed precision training;
* checkpointing;
* loss logging;
* retrieval-based evaluation;
* image-to-text and text-to-image Recall@K metrics.

> Note: This repository is currently a research prototype. The public version may require adapting configuration files and dataset paths before full training can be reproduced on a new machine.

## Evaluation

The model can be evaluated using retrieval-based metrics in the shared embedding space.

The current evaluation focuses on:

* **Image-to-text retrieval:** given a 3D neural activation map, retrieve the corresponding textual description.
* **Text-to-image retrieval:** given a textual description, retrieve the corresponding 3D neural activation map.

The trainer computes:

* Recall@1 for image-to-text retrieval;
* Recall@5 for image-to-text retrieval;
* Recall@1 for text-to-image retrieval;
* Recall@5 for text-to-image retrieval.

These metrics measure whether the correct paired text or image appears among the top retrieved candidates.

## Inference Example

After training, NeuroCLIP can be used to compare text and 3D neural activation maps in a shared embedding space.

```python
# Pseudocode example

image_features, _ = model.image_encoder(image_tensor)
text_features, _ = model.text_encoder_1(text_tokens)

image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)

similarity = image_features @ text_features.T
```

This enables retrieval of the most relevant scientific description for a neural activation map, or retrieval of the most relevant neural activation map for a given text query.

## Example Applications

NeuroCLIP can be used as a research prototype for:

* natural language-based retrieval of neural activation maps;
* cross-modal representation learning between scientific text and 3D neural data;
* interpretable exploration of biomedical or neuroscience datasets;
* studying alignment between textual descriptions and neural imaging patterns;
* building foundations for biomedical vision-language models.

## Limitations

This project is an experimental research prototype. Current limitations include:

* the full dataset is not publicly available;
* pretrained checkpoints are not currently included in the repository;
* dataset preparation depends on local metadata and activation-map files;
* the training pipeline is still being cleaned for reproducibility;
* evaluation has so far been conducted in an internal experimental setting;
* additional validation is needed before drawing strong scientific conclusions.

## Future Work

Future development will focus on:

* improving reproducibility and configuration management;
* adding clearer training and evaluation scripts;
* adding example notebooks for retrieval and visualization;
* adding pretrained checkpoints when possible;
* comparing 3D ResNet and 3D Vision Transformer encoders;
* improving local alignment between text tokens and neural activation regions;
* extending the model toward more robust biomedical and neuroscience retrieval tasks.

## Acknowledgements

This project was developed as part of my research on interpretable and efficient cross-modal learning for biomedical data.

## Author

**Van Dat Tran**

Research interests:

* Large Language Models;
* Multimodal Learning;
* Trustworthy AI;
* AI Systems;
* Deep Generative Models;
* Medical Image Analysis.

Email: [dat.tran220407@vnuk.edu.vn](mailto:dat.tran220407@vnuk.edu.vn)
GitHub: [trandat27jk](https://github.com/trandat27jk)

```
```
