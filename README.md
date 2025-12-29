# Vega-Blade-Life

## Predictive Maintenance for Industrial Blades

Estimate blade wear and predict replacement needs using physics-guided Φ-Clock features and a One-Class k-NN model.

---

## Dataset
- Original data (industrial blade signals, 1-year run) can be downloaded from:  
[Kaggle: One-Year Industrial Component Degradation](https://www.kaggle.com/datasets/inIT-OWL/one-year-industrial-component-degradation?resource=download)

---

## Project Files
- `notebook.ipynb` — Colab notebook with all preprocessing, training, and Gradio demo  
- `scaler.gz`, `knn.gz` — Saved models for scaling and k-NN inference  
- `params.json` — Thresholds and other parameters for inference  

---

## Method
1. Compute Φ = ∫|Torque| dt per cut  
2. Build feature vectors: [Φ, slope, acceleration]  
3. Train One-Class k-NN (99% healthy boundary)  
4. Real-time blade life estimation via Gradio dashboard

---

## Demo
Launch the interactive dashboard using Gradio to test blade conditions:

```python
import gradio as gr
# load scaler, knn, and threshold
# define `predict` function as in notebook
demo.launch(share=True)
