import numpy as np
import pickle 
import pytest
import os

from o2o.simulator import BivariateHawkesProcessSimulator

def test_default_parameters():
    """
    Test that default parameters (gamma, alpha, mu) are correctly initialized.
    """
    sim = BivariateHawkesProcessSimulator(M=1, T=1)
    assert sim.alpha.shape == (2,2)
    assert sim.gamma.shape == (2,2)
    assert sim.mu.shape == (1,2)

    np.testing.assert_allclose(sim.gamma, [[5.0, 1.0], [3.0, 2.0]], atol=1e-8)
    np.testing.assert_allclose(sim.alpha, [[0.6, 0.1], [0.2, 0.8]], atol=1e-8)
    np.testing.assert_allclose(sim.mu, [[0.1, 0.3]], atol=1e-8)


def test_no_events_for_zero_mu_and_alpha():
    """
    If mu and alpha are zero, the simulator should return no events at all
    """
    M = 2
    T = 5
    zero_mu = np.zeros((M, 2))
    zero_alpha = np.zeros((2,2))
    gamma = np.ones((2,2)) #The decay does not matter since if alpha vanishes and mu vanish the intensity=0
    sim = BivariateHawkesProcessSimulator(M, T, mu = zero_mu, alpha = zero_alpha, gamma = gamma)
    online_data, offline_data = sim.simulate()
    
    #There should be no events in either mark for any user
    assert isinstance(online_data, list) 
    assert isinstance(offline_data, list)
    assert len(online_data) == M 
    assert len(offline_data) == M
    for i in range(M):
        assert len(online_data[i]) == 0
        assert len(offline_data[i]) == 0

def test_reproducibility_with_fixed_seed():
    """
    Make sure that repeated runs with the same parameters produce identical output
    """

    sim1 = BivariateHawkesProcessSimulator(M = 2, T = 10)
    sim2 = BivariateHawkesProcessSimulator(M = 2, T = 10)
    online_data1, offline_data1 = sim1.simulate()
    online_data2, offline_data2 = sim2.simulate()

    assert online_data1 == online_data2
    assert offline_data1 == offline_data2


def test_baseline_only_events():
    """
    With alpha = 0 and mu > 0, only baseline Poisson events should be generated (no offspring events)
    """
    mu = np.array([[0.3, 0.2]])
    alpha = np.zeros((2,2))
    gamma = np.zeros((2,2))
    sim = BivariateHawkesProcessSimulator(M = 1, T = 10, mu = mu, alpha = alpha, gamma = gamma)
    online, offline = sim.simulate()
    assert len(online[0]) > 0 and len(offline[0]) > 0


def test_file_saving(tmp_path, monkeypatch):
    """
    Test that the simulator saves the data correctly
    """

    monkeypatch.chdir(tmp_path)
    sim = BivariateHawkesProcessSimulator(M = 2, T = 10)
    online, offline = sim.simulate(save_online_data = True, save_offline_data = True)
    assert os.path.isfile('online_data.pkl')
    assert os.path.isfile('offline_data.pkl')

    with open('online_data.pkl', 'rb') as file:
        online_data = pickle.load(file)
    with open('offline_data.pkl', 'rb') as file:
        offline_data = pickle.load(file)

    assert online_data == online
    assert offline_data == offline
