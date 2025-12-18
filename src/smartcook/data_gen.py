import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ==============================================================================
# SECTION 1: TIME-SERIES GENERATION (Legacy Support)
# ==============================================================================

def generate_cooking_session(session_id, food_type=None):
    """
    Simulates a full time-series cooking session (Temp/Humidity/Weight over time).
    """
    if food_type is None:
        food_type = np.random.choice(['chicken', 'steak', 'mystery_object'])
    
    time_steps = np.linspace(0, 60, 60)
    
    # --- PHYSICS ENGINE ---
    if food_type == 'chicken':
        temp_curve = 20 + (180 - 20) / (1 + np.exp(-0.15 * (time_steps - 15)))
        humidity_curve = 40 + 30 * np.exp(-0.01 * (time_steps - 20)**2)
        weight_curve = 1000 - 3 * time_steps 
        stages = np.zeros(60)
        stages[20:50] = 1 
        stages[50:] = 2   

    elif food_type == 'steak':
        temp_curve = 20 + (150 - 20) / (1 + np.exp(-0.25 * (time_steps - 10)))
        humidity_curve = 30 + 10 * np.exp(-0.01 * (time_steps - 10)**2)
        weight_curve = 300 - 1 * time_steps 
        stages = np.zeros(60)
        stages[10:30] = 1 
        stages[30:] = 2   

    elif food_type == 'mystery_object':
        temp_curve = 20 + (220 - 20) / (1 + np.exp(-0.4 * (time_steps - 5)))
        humidity_curve = 20 + 40 * np.exp(-0.1 * (time_steps - 5)**2)
        weight_curve = 500 - 0.5 * time_steps 
        stages = np.zeros(60)
        stages[5:15] = 1 
        stages[15:] = 2   

    # Add Noise
    temp_curve += np.random.normal(0, 2, 60)
    humidity_curve += np.random.normal(0, 1, 60)
    
    df = pd.DataFrame({
        "time": time_steps,
        "temperature": temp_curve,
        "humidity": humidity_curve,
        "weight": weight_curve,
        "stage_label": stages,
        "food_type": food_type, 
        "session_id": session_id
    })
    
    return df

class LabeledCookingDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.max_vals = torch.tensor([250.0, 100.0, 1000.0]) 

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        df = generate_cooking_session(session_id=idx)
        sensor_data = df[['temperature', 'humidity', 'weight']].values.astype(np.float32)
        x = torch.tensor(sensor_data) / self.max_vals
        label_stage = torch.tensor(df['stage_label'].iloc[-1], dtype=torch.long)
        target_temp = torch.tensor(df['temperature'].max(), dtype=torch.float32) / 250.0
        return x, label_stage, target_temp
    
class CookingDataset(LabeledCookingDataset):
    def __getitem__(self, idx):
        x, _, _ = super().__getitem__(idx)
        return x


# ==============================================================================
# SECTION 2: HIGH-FIDELITY SIMULATION (Hyper-Randomized)
# ==============================================================================

def generate_classification_mock_data():
    """ Generates data for Part 1: 'What is it?' """
    # Fresh RNG every time
    rng = np.random.default_rng() 
    
    # --- VISUAL PROOF OF RANDOMNESS ---
    # We add a "drift" variable. The clusters will move significantly every time.
    chicken_drift_x = rng.uniform(-0.5, 0.5)
    chicken_drift_y = rng.uniform(-0.5, 0.5)
    steak_drift_x = rng.uniform(-0.5, 0.5)
    
    # 1. Background Spectrum
    spec_x = rng.normal(0, 1.5, 500) 
    spec_y = rng.normal(0, 1.5, 500)
    
    # 2. Clusters (Now with Drift!)
    # Chicken centers around -2.0, plus random drift
    chicken_x = rng.normal(-2.0 + chicken_drift_x, 0.2, 20)
    chicken_y = rng.normal(0.0 + chicken_drift_y, 0.2, 20)
    
    # Steak centers around 0.9, plus random drift
    steak_x = rng.normal(0.9 + steak_drift_x, 0.15, 20)
    steak_y = rng.normal(-1, 0.15, 20)
    
    # Mystery Object (Jitters around)
    mystery_x = [1.0 + rng.uniform(-0.2, 0.2)]
    mystery_y = [1.0 + rng.uniform(-0.2, 0.2)]
    
    # 3. Waveforms (With amplitude variance)
    time_axis = np.linspace(0, 1, 100)
    # Chicken amplitude varies between 1.5 and 2.5 randomly
    amp_c = 2.0 + rng.uniform(-0.5, 0.5) 
    chicken_profile = amp_c * np.sin(3 * np.pi * time_axis) + rng.normal(0, 0.1, 100)
    
    steak_profile = 8 * np.sin(3 * np.pi * time_axis) + rng.normal(0, 0.1, 100)
    mystery_profile = 5 * np.sin(6 * np.pi * time_axis) + 2 + rng.normal(0, 0.1, 100)
    
    return {
        "latent": {
            "spectrum": (spec_x, spec_y),
            "chicken": (chicken_x, chicken_y),
            "steak": (steak_x, steak_y),
            "mystery": (mystery_x, mystery_y)
        },
        "waveforms": (time_axis, chicken_profile, steak_profile, mystery_profile)
    }

def generate_physics_simulation_data(n_samples=50):
    """
    Generates High-Fidelity Data.
    Hyper-Randomized: Even the physics parameters jitter slightly.
    """
    rng = np.random.default_rng() 

    # Physics "Law" Drift (Simulating slightly different ambient conditions)
    base_time = 10 + rng.uniform(-1, 1) # Sometimes base time is 9, sometimes 11

    def physics_law(density):
        return base_time + (density ** 1.5) * 1.5

    # --- Generate Random Background Data ---
    chicken_density = rng.normal(2.0, 0.3, n_samples)
    chicken_time = physics_law(chicken_density) + rng.normal(0, 2, n_samples)

    steak_density = rng.normal(6.0, 0.5, n_samples)
    steak_time = physics_law(steak_density) + rng.normal(0, 3, n_samples)

    spectrum_density = rng.uniform(1.0, 8.0, 500)
    spectrum_time = physics_law(spectrum_density) + rng.normal(0, 4, 500) 

    bone_density = rng.normal(10.0, 0.5, 10) 
    bone_time = physics_law(np.full(10, 6.0)) + rng.normal(0, 2, 10) 
    
    frozen_density = rng.normal(3.5, 0.2, 10)
    frozen_time = physics_law(frozen_density) + 8 + rng.normal(0, 1, 10) 

    # --- Mystery Object (High Variance) ---
    true_density = 7.5
    noisy_input = true_density + rng.normal(0, 0.5) # Increased noise
    mystery_input = np.array([[noisy_input]])

    return {
        "train": {
            "density": np.concatenate([spectrum_density, bone_density, frozen_density]),
            "time": np.concatenate([spectrum_time, bone_time, frozen_time])
        },
        "plotting": {
            "spectrum": (spectrum_density, spectrum_time),
            "chicken": (chicken_density, chicken_time),
            "steak": (steak_density, steak_time),
            "bone": (bone_density, bone_time),
            "frozen": (frozen_density, frozen_time)
        },
        "mystery": mystery_input
    }