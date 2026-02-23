# Hyperspectral-3DCNN-Transfer-Learning
End-to-end 3D-CNN pipeline for hyperspectral crop classification. Uses PCA and Transfer Learning to tackle the Curse of Dimensionality and apply models across different geographic areas.

## The Problem
Standard RGB images have 3 channels of light. Hyperspectral sensors capture hundreds of narrow, continuous spectral bands, with over 270 bands available. This data shows the detailed chemical and physical properties of plants. However, directly inputting over 270 channels into a neural network leads to significant redundancy, gradient explosion, and memory failure. Additionally, models trained in one geographic area often do not perform well in new areas because of atmospheric differences, such as varied sun angles, humidity, and soil backgrounds.

## The Architecture
This pipeline addresses these issues with a 4-phase mathematical engine:

1. **Spectral Compression (PCA):** It reduces the raw 270-band hypercube to the top 30 Principal Components. This process isolates the most important chemical signatures and whitens the data for neural network stability.
2. **Spatial-Spectral Extraction:** This is a custom algorithm that pads and slices to extract microscopic `(5, 5, 30)` 3D geometric tensors. This provides the AI with spatial context.
3. **Base 3D Convolutional Neural Network:** This uses a custom Keras `Conv3D` architecture trained from scratch on the **LongKou** dataset, which includes 9 unique crop classes. The 3D kernels learn how light reflects.
4. **Transfer Learning & Fine-Tuning:** The model's "head" is removed and replaced with a new 16-class output layer for processing the **HanChuan** dataset. The 340,000 parameter "brain" is briefly frozen to map the new classes. It is then unfrozen with a very small learning rate (`1e-5`) to fine-tune the physical weights to the new atmosphere, preventing major loss of previous knowledge.

## Dataset
This pipeline uses the **WHU-Hi Hyperspectral Dataset** from `kagglehub`.  
* **Base Domain:** WHU-Hi-LongKou (270 bands, 9 classes).  
* **Transfer Domain:** WHU-Hi-HanChuan (274 bands, 16 classes).

## Results
- By using Transfer Learning and Fine-Tuning, the model adjusts to a new agricultural region with 7 previously unseen crops. It improves from random guessing to over **~91% Validation Accuracy**.

- The last phase of the pipeline goes through the spatial coordinates of the farm to create a 2D predictive map. The fine-tuning process effectively removes "salt-and-pepper" noise and clarifies the boundaries between different crop fields.

**Requirements:**
`tensorflow`, `scikit-learn`, `numpy`, `scipy`, `matplotlib`, `kagglehub`
