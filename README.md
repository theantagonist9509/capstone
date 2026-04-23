# Visualizing Medical Classifier Decision Boundaries via Latent Space Traversal
## Brief Description
We use autoencoders to map the latent space of input images to the medical classifier's outputs. We explore the classifier's decision boundary by moving through this space along two axes:

1. The Gradient Axis: We move with or against the gradient to see which image features increase or decrease the classifier's score.

2. The PCA Axis: We move sideways using PCA to find different-looking images that yield the same score.

## Problem Statement Details
### Motivation
The goal is to build a tool that helps clinicians decide whether to trust a classifier's prediction on a specific image. By encoding a skin lesion image into a compact latent space, we can move through that space in directions that hold the classifier's confidence constant or change it towards a decision boundary. Decoding those points back into images shows what can change while the prediction stays the same, and where it breaks. The core output is a "boundary image" - the closest plausible image at which the classifier flips its prediction. A clinician can look at this image and judge whether the flip makes sense visually.

### Dataset
We initially targeted ISIC 2020 (33k images, biopsy-confirmed binary labels), but the malignant class was only 1.7% of the data. Even with weighted loss, the classifier overfit the malignant training samples and did not generalize. Thus, we switched to HAM10000 (ISIC 2018), which has 10,015 images across 7 classes. Collapsing to binary gives a more tractable 20% malignant rate.

We restricted training to melanoma (MEL, n=1113) vs melanocytic nevus (NV, n=6705). This is because a binary classification problem is easier to study than a multiclass one. Furthermore, the ABCD criterion used by dermatologists is specifically designed for differentiating NV from MEL, and is simple enough that non-professionals can understand and interpret the traversal results visually.

To address class imbalance, we use `BCEWithLogitsLoss` with `pos_weight` and weighted random sampling during training. For data augmentation, we apply flips, rotations, and colour jitter to ensure robustness.

## Delivered Components
### 1. Models
We provide PyTorch implementations for our core architectures in `models.py`, which include an EfficientNet-B0 based binary classifier for NV vs MEL, alongside both standard Autoencoder (AE) and Variational Autoencoder (VAE) architectures using symmetric `ConvTranspose2d` decoders.

### 2. Training Pipelines
Complete training scripts are available for both the classifier (`playground/efficientnet_nv_mel_classifier.py`) and the autoencoder/VAE (`playground/efficientnet_nv_mel_ae.py`). These pipelines feature weighted random sampling to handle class imbalance, extensive data augmentation, label smoothing, and automatic per-epoch checkpointing. Custom perceptual and structural losses (VGG and MS-SSIM) are defined in `losses.py`.

### 3. Evaluation Pipelines
To ensure the autoencoder preserves features relevant to the classifier, we evaluate the classifier's performance on both original images and autoencoder reconstructions. The `playground/recon_cls_perf.py` script computes comprehensive metrics (AUC, Precision, Recall, F1, Confusion Matrix) on the reconstructed data to validate the integrity of the latent space.

### 4. Utilities
Our system includes robust dataset handlers in `datasets.py` with support for lazy-loading and dynamic transformations. Additional utilities handle model checkpoint loading (`utils.py`), orthogonal PCA basis calculations for traversal (`utils.py`), and the generation of entirely synthetic datasets from autoencoder reconstructions (`playground/recon_dataset.py`).

### 5. Interactive GUIs for Latent Space Traversal
We provide interactive web-based tools to visually explore the latent space. `playground/efficientnet_nv_mel_latent_traversal_gui.py` allows users to traverse along gradient or PCA axes, or interpolate between images, observing how the reconstructed image and classifier confidence change in real-time.

## Analysis and Iterations
### 1. Classifier Model
- **Initial State**: Our baseline classifier exhibited a heavily bimodal output distribution.
- **Iteration 1**: Implemented label smoothing to address this. While predictions remained somewhat bimodal, the probabilities for both classes no longer exceeded those of the smoothed labels, keeping the logits closer together.
- **Future Work**: A potential future improvement is to use stochastic labels (adding random noise to the targets, e.g., $0 \pm \epsilon$, $1 \pm \epsilon$) to further prevent the model from becoming overconfident.

### 2. Autoencoder (AE) Model
- **Initial State**: Early reconstructions from the AE were excessively blurry. This severely degraded the classifier's performance on reconstructed images, which is problematic because our traversal method relies on backpropagating through the AE.
- **Iteration 1**: Experimented with VGG (perceptual) loss. This yielded better classifier performance on reconstructions, but the results still fell short.
- **Iteration 2**: Tested MS-SSIM loss, which outperformed VGG loss, though room for improvement remains.
- **Iteration 3**: Attempted to use a larger classifier (EfficientNet-B2) to compensate, but it yielded worse test performance even with a dropout of 0.4, indicating insufficient data.
- **Current Resolution**: Due to time and compute constraints, we decided to fine-tune the classifier directly on the AE reconstructed data. This bounds the experiment to a slightly weaker result but serves as a solid sanity check to validate the soundness of our research direction.
- **Future Work**: Future architectural iterations might combine both VGG and MS-SSIM losses or explore GAN-style architectures to produce sharper images.

### 3. Traversal
- **Initial State**: During latent space traversal, the AE manifold was not entirely smooth, leading to non-monotonic classifier confidence along the gradient direction and non-uniform step sizes. The generated images also often appeared synthetic with artificial-looking color artifacts.
- **Iteration 1 (Smoothness)**: Switched from an AE to a VAE. This improved manifold smoothness but came at the cost of worse reconstruction quality.
- **Iteration 2 (Artifacts)**: Attempted to mitigate color artifacts by training on grayscale images. However, this degraded classifier performance compared to RGB training.
- **Iteration 3 (Orthogonal Traversal)**: To fix the synthetic artifacts, we introduced an orthogonal setting to the traversal: moving in a direction orthogonal to the gradient of the AE's reconstruction loss. The intuition is that synthetic-looking decodes are out-of-distribution and incur a large gradient in reconstruction loss; avoiding this gradient keeps images realistic.
- **Iteration 4 (Visualizations)**: Because the orthogonal change made visual differences between original and traversed images less pronounced, we added a gamma-scaled luminance difference map overlaid on the original image to clearly highlight subtle changes.
- **Iteration 5 (Interpolation)**: Added a validation-set image interpolation feature to make the latent space more transparent. This allows us to observe smooth changes in classifier confidence and uses Grad-CAM to reveal exactly which regions are driving the classifier's predictions along the interpolation path.

## Code Structure
Our codebase is organized to separate core definitions from executable scripts:

- **`models.py`**: Contains the PyTorch architecture definitions for the EfficientNet-B0 based binary classifier, the standard Autoencoder (AE), and the Variational Autoencoder (VAE).
- **`datasets.py`**: Houses the dataset classes (e.g., `ISIC2018Dataset`, `TransformDataset`) used for loading and augmenting the skin lesion images.
- **`losses.py`**: Implements custom loss functions to improve reconstruction quality, such as `VGGLoss` (perceptual loss) and `MS_SSIMLoss` (structural similarity).
- **`utils.py`**: Provides shared helper functions, including utilities for checkpoint loading and calculating orthogonal PCA bases for latent traversal.
- **`playground/`**: This directory contains all the standalone executable scripts for training, evaluation, and visualization:
  - **`efficientnet_nv_mel_classifier.py`**: Training pipeline for the classifier, incorporating label smoothing and class balancing.
  - **`efficientnet_nv_mel_ae.py`**: Training pipeline for the autoencoders, utilizing custom reconstruction losses.
  - **`recon_cls_perf.py`**: Evaluation script to measure the classifier's performance specifically on images reconstructed by the autoencoder.
  - **`recon_dataset.py`**: Script to generate and save synthetic datasets by running the validation set through the trained autoencoder.
  - **`efficientnet_nv_mel_latent_traversal_gui.py`**: The interactive Gradio-based web interface for exploring the latent space, offering gradient traversal, PCA axis traversal, and image interpolation with Grad-CAM overlays.
