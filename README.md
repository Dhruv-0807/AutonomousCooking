# Autonomous Cooking: Physics-based AI for Universal Food Classification

This repository contains work for the project **"Universal Physics Spectrum and Anomaly Detection for Autonomous Cooking."**  
It builds predictive models for food classification and cooking state stability from multi-sensor physics-based time-series and studies how well these models generalise across various food categories.

---

## 1. Dataset

- **Source:** Internal Physics-based sensor data and universal food spectral logs.
- Cycle-based time-series from multiple sensors:
  - Thermal profiles, moisture retention levels, density markers, and phase-transition indicators.
- Three primary labels per cycle:
  - **Food_Category**, **Cooking_State**, **Anomaly_Flag (0/1)**.

---

## 2. Methodology

1. **Data preparation**
   - Load raw physics sensor streams, align cooking cycles, and join with ground-truth labels into a master data structure.

2. **Feature engineering**
   - For every sensor & cycle, compute compact **time-domain** and **frequency-domain** features  
     (mean/std, spectral energy, thermal diffusivity, and moisture band powers).
   - Concatenate features to form a "Universal Physics Spectrum" tabular matrix.

3. **Modelling Pipeline**
   - **Pretraining** (`2_pretraining.ipynb`): Trains a physics-informed encoder to learn robust representations of food signatures that generalize across kitchen environments.
   - **Downstream Tasks** (`3_downstreamtasks.ipynb`): Fine-tunes the model for specific ingredient classification (e.g., protein vs. vegetable).
   - **Baselines:** Linear regression and single-feature curve fits for temperature/doneness.

4. **Unsupervised analysis**
   - KMeans and DBSCAN on feature space to identify hidden regimes (e.g., "incipient_burning" state).
   - Correlation and permutation-importance analyses to determine the most influential physical sensors for cooking stability.

---

## 3. Key Results & Insights

### 3.1 Predictive performance

- **Cooking_State**
  - Highly predictable using single-feature thermal curve fits, reaching **R² ≈ 0.99**.
- **Food_Category**
  - The physics-based encoder provides high accuracy in identifying food states even when visual conditions are obscured (e.g., steam or low light).
- **Anomaly_Detection**
  - The system can successfully distinguish between a standard cooking process and a failure state (Anomaly) by identifying deviations in the expected physics waveform.
  - **Tree-based models on features perform significantly better** than raw sequence models, suggesting the engineered physics features capture the necessary temporal structure for error detection.

### 3.2 Feature importance & regimes

- Permutation importance for **Anomaly_Detection** shows that:
  - Higher-order statistics of **thermal diffusivity** and **moisture band powers** are more influential than raw mean temperature levels.
- **Clustering analysis** successfully isolates:
  - One large "optimal" cooking cluster and a smaller, noisy cluster corresponding to failure states.
  - Successfully captures cycles in an **incipient degradation regime** (e.g., initial burning) rather than full failure.

---

## 4. Execution Sequence

To replicate the results, notebooks must be run in the following order:

1. `1_exploration.ipynb`: Initial data visualization and sensor alignment.
2. `2_pretraining.ipynb`: Model training and weight generation.
3. `3_downstreamtasks.ipynb`: Performance evaluation and classification.
4. `4_preferences.ipynb`: Configuration of user thresholds and taste profiles.
5. `5_anomaly_detection.ipynb`: Final system testing and regime discovery.

---

## 5. Skills Demonstrated

- Physics-Informed Machine Learning (PINNs) and universal spectral analysis
- Time-series signal processing and feature generation for sensor-heavy environments
- Permutation importance, clustering, and regime discovery for model interpretation
- Anomaly detection in dynamic, non-linear systems
- End-to-end ML Pipeline design, from raw data to a deployable encoder
