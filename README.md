# Visualizing Medical Classifier Decision Boundaries via Latent Space Traversal
## Brief Description
We use autoencoders to map the latent space of input images to the medical classifier's outputs. We explore the classifier's decision boundary by moving through this space along two axes:

1. The Gradient Axis: We move with or against the gradient to see which image features increase or decrease the classifier's score.

2. The PCA Axis: We move sideways using PCA to find different-looking images that yield the same score.

## Motivation & Problem Statement
### Dataset

## Delivered Components

## Analysis and Iterations


## Code Structure




- components
    - motivation & problem statement
    - problem statement details (dataset)
        - chain of reasoning to arrive at isic2018, from isic2020
        - using only nv vs mel
            - binary is easier to study than multiclass
            - abcd criterion used by dermatologists is for nv vs mel, is simple enough that someone who is not a trained professional may be able to understand and interpret
        - other considerations
            - class balancing
            - data augmentation (or lack thereof)
            - preprocessing
    - delivered
        - models
            - classifiers
            - aes
        - training pipelines
        - evaluation pipelines
            - classifier
                - on original images
                - on aes reconstructions
        - utilities
            - checkpointing
            - synthetic dataset generation from aes
            - automatic latent space traversal script

        - interactive gui for latent space traversal
