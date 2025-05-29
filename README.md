# Microstructure Analysis of Ti-6Al-4V Alloy Using UNet and Diffusion Models

## Overview  
This repository contains code for analyzing Ti-6Al-4V alloy microstructure images using advanced deep learning techniques. The pipeline preprocesses SEM images, splits them into sub-images to augment data, and employs a UNet-based diffusion model to generate synthetic microstructure images and evaluate their quality.

## Objective  
The primary objective is to analyze Ti-6Al-4V microstructure images and their corresponding labels by dividing each image into two parts, effectively increasing the dataset size. Using a UNet-based diffusion model, we aim to:

- Generate realistic synthetic microstructure images  
- Evaluate model performance using perceptual similarity metrics (e.g., LPIPS)  
- Identify which microstructural features or image categories yield the best metric performance  
- Visualize results via heatmaps to understand model suitability across different microstructure types

In essence, this study seeks to determine for which image categories or features the diffusion model performs optimally.

## Results & Analysis
Model performance was quantitatively evaluated using the Learned Perceptual Image Patch Similarity (LPIPS) metric across various microstructure textures. Each image was classified based on its texture (e.g., lamellar, martensitic, basketweave), and the LPIPS scores were aggregated for both left and right halves to increase resolution in texture-wise analysis.

The LPIPS comparisons were visualized using heatmaps, where each row corresponded to a specific microstructure texture. Lower LPIPS values (cooler colors) indicated better perceptual fidelity between generated and real images.

The analysis revealed:

- Texture-dependent variation in LPIPS scores, with certain textures consistently showing better alignment between real and generated samples.

- Model-specific strengths and weaknesses, as some architectures produced higher-quality images for certain textures but underperformed for others.

- Regional performance trends, which highlighted specific microstructure regions that were more challenging for the model to replicate.

These insights are instrumental in guiding model selection, architecture tuning, and identifying microstructural features that benefit most from synthetic augmentation.

## Scope  
- Image preprocessing and augmentation for microstructure datasets  
- Implementation of UNet architecture integrated within a diffusion model framework  
- Quantitative assessment of model-generated image quality using advanced perceptual metrics  
- Visualization and interpretation of metric distributions across microstructural feature classes

## Applications  
- Accelerating microstructure characterization in materials science  
- Enhancing synthetic data generation for training robust AI models  
- Informing model selection based on feature-specific performance  
- Supporting alloy design and process optimization through improved microstructure analysis

## Usage  
1. Place Ti-6Al-4V SEM images and their labels in the `input/` directory.  
2. Execute the main processing and model training script:  
   ```bash
   python code.py
3. Outputs including synthetic images, evaluation metrics, and heatmaps will be saved in the output/ directory.

## Contributing
Contributions are encouraged. Please open an issue or submit a pull request for improvements or bug fixes.
