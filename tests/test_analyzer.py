import numpy as np
import pandas as pd
import pytest
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from o2o.analyzer import O2OAnalyzer

# Sample fixture data for tests
@pytest.fixture
def sample_data():
    data_online = [np.array([0.5, 1.0, 1.5]), np.array([0.6,0.8,0.9]), np.array([1.2,1.3]), np.array([1.5]),
            np.array([1.2,1.3,1.9])]
    data_offline = [np.array([0.7, 1.8]), np.array([0.1, 0.5]), np.array([1.8]), np.array([1.2,1.5]),
            np.array([1.2])]
    return data_online, data_offline


# Sample fake fit fot test
@pytest.fixture
def sample_fit():
    # Mock fit: shapes should match Stan output conventions in your class
    return {
        'mu': np.ones((100, 5, 2)),
        'alpha': np.ones((100, 2, 2)),
        'gamma': np.ones((100, 2, 2)),
    }



def test_sample_size_table(sample_data, sample_fit, tmp_path, monkeypatch):
    """
    Test that the sample_size() method returns correct counts and optionally saves to CSV
    """
    data_online, data_offline = sample_data
    M = 5
    T = 2
    analyzer = O2OAnalyzer(data_online, data_offline, M, T, fit = sample_fit)
    table = analyzer.sample_size()
    assert isinstance(table, pd.DataFrame)
    assert list(table['Online sample size']) == [3,3,2,1,3]
    assert list(table['Offline sample size']) == [2,2,1,2,1]

    #Test saving
    monkeypatch.chdir(tmp_path)
    output = analyzer.sample_size(save_sizes = True)
    assert os.path.exists('sizes.csv')
    saved = pd.read_csv('sizes.csv')
    assert 'Online sample size' in saved.columns
    assert 'Offline sample size' in saved.columns


def test_plot_intensity_runs_without_error(sample_data, sample_fit, tmp_path, monkeypatch):
    """
    Test that plot_intensity runs correctly
    """

    data_online, data_offline = sample_data
    M = 5
    T = 2

    analyzer = O2OAnalyzer(data_online, data_offline, M, T, sample_fit)
    monkeypatch.chdir(tmp_path)
    #Should not raise any error and should save EPS if save_fige = True
    analyzer.plot_intensity(save_figs = True)
    #list all the created EPS file, and check that the number of created EPS files = the number of users
    eps_files = [f for f in os.listdir('.') if f.endswith('.eps')]
    assert len(eps_files) == M

def test_non_array_input():
    """
    If someone passes lists of lists instead of lists of numpy arrays, the function should raise error or 
    convert the data to numpy
    """
    fit = {
            'mu' : np.ones((2, 2, 100)),
            'alpha' : np.ones((2,2,100)),
            'gamma' : np.ones((2,2,100))
            }
    data_online = [[0.1,0.2], [0.3,0.4]]
    data_offline = [[0.5, 0.6], [0.7, 0.9]]

    analyzer = O2OAnalyzer(data_online, data_offline, M = 2, T = 1, fit  = fit)
    table = analyzer.sample_size()
    assert list(table['Online sample size']) == [2 , 2]
    assert list(table['Offline sample size']) == [2, 2]
