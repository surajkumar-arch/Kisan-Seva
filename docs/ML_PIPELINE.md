# EDAPHNET Machine Learning Pipeline Specification

## 1. Dual-Branch Vision Pipeline Architecture

### 1.1 Soil Branch
```
Soil Image -> Quality Gate -> EfficientNet-B0 -> Logits -> Temperature Scaling (T=1.12) -> Mahalanobis OOD Detector -> Calibrated Soil Prediction
```

- **Backbone**: EfficientNet-B0 with ImageNet feature weights.
- **Latent Embedding**: 1280-dimensional feature vector extracted after global average pooling.
- **Classes**: Alluvial, Black, Clay, Red, Sandy, Loamy.
- **Calibration**: Temperature scaling minimizes negative log-likelihood (NLL) on validation logits.
- **OOD Detection**: Mahalanobis distance in 1280D embedding space against class-conditional centroids with tied inverse covariance. Cutoff threshold: $D_M > 16.5$.

### 1.2 Foliage Pathology Branch
```
Leaf Image -> Quality Gate -> EfficientNet-B0 -> Logits -> Temperature Scaling -> Reliability Gate -> Foliar Pathology Prediction
```

- **Backbone**: EfficientNet-B0.
- **Localization**: YOLOv8/YOLO11 regression head extracting bounding boxes around necrotic spots with non-maximum suppression (NMS) at IoU 0.45.
- **Segmentation**: PyTorch U-Net architecture producing pixel-level binary masks to measure damaged surface area percentage relative to total leaf pixels.

---

## 2. Confidence Calibration & Temperature Scaling

Standard deep neural networks tend to be overconfident when optimized with cross-entropy loss. We employ post-hoc **Temperature Scaling**:

$$\hat{p}_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

Where:
- $z_i$ is the unnormalized logit for class $i$.
- $T > 0$ is the learned temperature parameter optimized via L-BFGS on validation data.
- $T$ does not change the argmax prediction, preserving top-1 accuracy while reducing Expected Calibration Error (ECE).

---

## 3. Out-of-Distribution (OOD) Detection

Raw softmax probabilities are fundamentally unsuited for out-of-distribution detection. EDAPHNET implements **Mahalanobis Distance in Embedding Space**:

$$D_M(x) = \min_c \sqrt{(\phi(x) - \mu_c)^T \Sigma^{-1} (\phi(x) - \mu_c)}$$

Where:
- $\phi(x)$ is the 1280-dimensional latent embedding.
- $\mu_c$ is the centroid of class $c$.
- $\Sigma^{-1}$ is the tied empirical inverse covariance matrix regularized with shrinkage.

### Decision Matrix
| Condition | Decision | Reliability Level | Machine-Readable Reason |
|:---|:---|:---|:---|
| $D_M > 16.5$ and Confidence $< 0.50$ | **ABSTAIN** | `ABSTAIN` | `OOD_DETECTED` |
| Image Quality Fails | **ABSTAIN** | `ABSTAIN` | `IMAGE_QUALITY_REJECTED` |
| $D_M > 16.5$ or Confidence $< 0.40$ | **BEST_EFFORT** | `LOW` | `LOW_CONFIDENCE` |
| Laboratory N-P-K Missing | **PROCEED** | `MODERATE` | `NONE` |
| In-Distribution & Calibrated | **PROCEED** | `HIGH` | `NONE` |

---

## 4. Explainable AI (XAI)

- **Grad-CAM**: Computes the gradient of the winning class score $y^c$ with respect to the feature map activations $A^k$ of the final convolutional layer:
  $$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}$$
  $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
- **SHAP Feature Attributions**: For the tabular crop recommendation model, local marginal contributions quantify positive and negative drivers for each crop recommendation (e.g. +Rainfall, +Soil Type, -Phosphorus).
