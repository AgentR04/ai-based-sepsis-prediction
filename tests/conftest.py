"""
Pytest configuration and shared fixtures for sepsis prediction tests
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_cohort_data():
    """Sample cohort data for testing"""
    return pd.DataFrame({
        'icustay_id': [1001, 1002, 1003, 1004, 1005],
        'subject_id': [10, 20, 30, 40, 50],
        'hadm_id': [100, 200, 300, 400, 500],
        'gender': ['M', 'F', 'M', 'F', 'M'],
        'age': [45, 62, 38, 71, 55],
        'admission_type': ['EMERGENCY', 'URGENT', 'EMERGENCY', 'ELECTIVE', 'URGENT'],
        'intime': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05']),
        'outtime': pd.to_datetime(['2020-01-03', '2020-01-05', '2020-01-06', '2020-01-07', '2020-01-08']),
        'icu_los_hours': [48, 72, 72, 72, 72]
    })


@pytest.fixture
def sample_vitals_data():
    """Sample vital signs data for testing"""
    icustays = [1001, 1001, 1001, 1002, 1002]
    hours = [0, 1, 2, 0, 1]
    
    data = []
    for icu, hour in zip(icustays, hours):
        for vital in ['heart_rate', 'map', 'resp_rate', 'temperature', 'spo2']:
            data.append({
                'icustay_id': icu,
                'hour': hour,
                'vital': vital,
                'value': np.random.uniform(60, 100) if vital != 'temperature' else 37.0
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_labs_data():
    """Sample laboratory data for testing"""
    icustays = [1001, 1001, 1001, 1002, 1002]
    hours = [0, 1, 2, 0, 1]
    
    data = []
    for icu, hour in zip(icustays, hours):
        for lab in ['creatinine', 'bilirubin', 'platelets', 'lactate']:
            value = np.random.uniform(0.5, 2.0) if lab != 'platelets' else 150
            data.append({
                'icustay_id': icu,
                'hour': hour,
                'lab': lab,
                'value': value
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_features_data():
    """Sample feature data with rolling statistics"""
    np.random.seed(42)
    n_rows = 100
    
    data = {
        'icustay_id': np.repeat([1001, 1002, 1003, 1004], 25),
        'hour': np.tile(np.arange(25), 4),
        'heart_rate': np.random.uniform(60, 100, n_rows),
        'map': np.random.uniform(65, 90, n_rows),
        'resp_rate': np.random.uniform(12, 20, n_rows),
        'temperature': np.random.uniform(36.5, 38.5, n_rows),
        'spo2': np.random.uniform(90, 100, n_rows),
        'creatinine': np.random.uniform(0.5, 2.0, n_rows),
        'bilirubin': np.random.uniform(0.5, 1.5, n_rows),
        'platelets': np.random.uniform(100, 300, n_rows),
        'lactate': np.random.uniform(0.5, 2.5, n_rows),
        'age': np.repeat([45, 62, 38, 71], 25),
        'gender': np.repeat([1, 0, 1, 0], 25)  # M=1, F=0
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_sofa_data():
    """Sample data with SOFA scores"""
    np.random.seed(42)
    n_rows = 100
    
    data = {
        'icustay_id': np.repeat([1001, 1002, 1003, 1004], 25),
        'hour': np.tile(np.arange(25), 4),
        'platelets': np.random.uniform(50, 250, n_rows),
        'bilirubin': np.random.uniform(0.5, 3.0, n_rows),
        'map': np.random.uniform(60, 85, n_rows),
        'creatinine': np.random.uniform(0.8, 3.0, n_rows),
        'sofa_platelets': np.random.randint(0, 3, n_rows),
        'sofa_bilirubin': np.random.randint(0, 2, n_rows),
        'sofa_map': np.random.randint(0, 2, n_rows),
        'sofa_creatinine': np.random.randint(0, 3, n_rows),
        'sofa_total': np.random.randint(0, 8, n_rows)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_train_dataset():
    """Sample training dataset with labels"""
    np.random.seed(42)
    n_rows = 200
    
    data = {
        'icustay_id': np.repeat([1001, 1002, 1003, 1004, 1005], 40),
        'hour': np.tile(np.arange(40), 5),
        'heart_rate': np.random.uniform(60, 120, n_rows),
        'map': np.random.uniform(50, 100, n_rows),
        'resp_rate': np.random.uniform(10, 30, n_rows),
        'temperature': np.random.uniform(36, 39, n_rows),
        'spo2': np.random.uniform(85, 100, n_rows),
        'creatinine': np.random.uniform(0.5, 4.0, n_rows),
        'bilirubin': np.random.uniform(0.3, 5.0, n_rows),
        'platelets': np.random.uniform(30, 350, n_rows),
        'lactate': np.random.uniform(0.5, 4.0, n_rows),
        'age': np.repeat([45, 62, 38, 71, 55], 40),
        'gender': np.repeat([1, 0, 1, 0, 1], 40),
        'label': np.random.randint(0, 2, n_rows)
    }
    
    # Make imbalanced (realistic)
    data['label'] = np.where(np.random.random(n_rows) < 0.85, 0, 1)
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directories"""
    data_dir = tmp_path / "data"
    (data_dir / "raw").mkdir(parents=True)
    (data_dir / "features").mkdir(parents=True)
    (data_dir / "labels").mkdir(parents=True)
    (data_dir / "model").mkdir(parents=True)
    
    return data_dir
