import numpy as np
import pytest
import pandas as pd
import pickle
from o2o.estimator import ParameterEstimator

def test_estimator_initialization():
    """
    Test estimator initializes and stores the data correctly
    """
    data_online = [np.array([0.5, 1.2, 1.6]), np.array([2.3, 2.4])]
    data_offline = [np.array([1.8, 1.9, 2.8]), np.array([[1.1]])]
    M = 2
    T = 3
    est = ParameterEstimator(data_online, data_offline, M, T)
    assert len(est.data_online) == 2
    assert len(est.data_offline) == 2


def test_spillover_and_decays_tables():
    """
    Sanity check on the spillover and decay rate tables with mocked fit
    """
    data_online = [np.array([1.2, 1.3, 1.6, 1.9])]
    data_offline = [np.array([1.8,1.9])]

    M = 1
    T = 2

    est = ParameterEstimator(data_online, data_offline, M, T)
    #Mock fit with arrays shaped in the format of Stan output, say we will do 100 sample (last axis)
    est.fit = {
            'alpha' : np.ones((2,2,100)),
            'gamma' : np.ones((2,2,100)),
            'mu' : np.ones((1,2, 100))
            }
    alpha_df, gamma_df = est.spillover_and_decays_values_and_CI()
    assert isinstance(alpha_df, pd.DataFrame)
    assert isinstance(gamma_df, pd.DataFrame)
    
    for df in [alpha_df, gamma_df]:
        assert 'Effect' in df.columns
        assert 'Value' in df.columns
        assert '95% CI' in df.columns

def test_baseline_and_tables():
    """
    Sanity check for the baseline intensity and CI for each user online and offline table
    """
    data_online = [np.array([1.2, 1.3, 1.6, 1.9])]
    data_offline = [np.array([1.8,1.9])]

    M = 1
    T = 2

    est = ParameterEstimator(data_online, data_offline, M, T)
    #Mock fit with arrays shaped in the format of Stan output, say we will do 100 sample (last axis)
    est.fit = {
            'alpha' : np.ones((2,2,100)),
            'gamma' : np.ones((2,2,100)),
            'mu' : np.ones((1,2, 100))
            }
    base_df = est.base_and_CI()
    assert isinstance(base_df, pd.DataFrame)
    assert 'User' in base_df.columns
    assert 'Online baseline intensity' in base_df.columns
    assert 'Offline baseline intensity' in base_df.columns

def test_percentage_tables():
    """
    Sanity checks on the table where there are the percentage of effect that each event has on the other
    """
    data_online = [np.array([1.2, 1.3, 1.6, 1.9])]
    data_offline = [np.array([1.8,1.9])]

    M = 1
    T = 2

    est = ParameterEstimator(data_online, data_offline, M, T)
    #Mock fit with arrays shaped in the format of Stan output, say we will do 100 sample (last axis)
    est.fit = {
            'alpha' : np.ones((2,2,100)),
            'gamma' : np.ones((2,2,100)),
            'mu' : np.ones((1,2, 100))
            }
    percentage_df = est.spillover_percentage()
    assert isinstance(percentage_df, pd.DataFrame)
    assert 'User' in percentage_df.columns
    assert '% Online  → Offline' in percentage_df.columns
    assert '% Offline → Online' in percentage_df.columns

def test_mismach_error_raise():
    data_online = [np.array([1.0])]
    data_offline = [np.array([2.0])]
    T = 2
    M = 3
    with pytest.raises(ValueError, match = "data should match M"):
        ParameterEstimator(data_online, data_offline, M, T)

def test_fit_pickling(tmp_path, monkeypatch):
    """
    Test that the estimator's fit dictionary can be correctly pickled and unpickled without data loss.
    """
    data_online = [np.array([1.2, 1.3, 1.6, 1.9])]
    data_offline = [np.array([1.8,1.9])]

    M = 1
    T = 2

    monkeypatch.chdir(tmp_path)
    est = ParameterEstimator(data_online, data_offline, M, T)


    fake_fit = {
            'alpha' : np.ones((2,2,100)),
            'gamma' : np.ones((2,2,100)),
            'mu' : np.ones((1,2,100))
            }
    est.fit = fake_fit
    #pickle it for testing then compare
    with open('fit.pkl', 'wb') as file:
        pickle.dump(fake_fit, file)
    #Load and compare
    with open('fit.pkl', 'rb') as file:
       loaded =  pickle.load(file)

    assert (loaded['alpha'] == fake_fit['alpha']).all()
    assert (loaded['gamma'] == fake_fit['gamma']).all()
    assert (loaded['mu'] == fake_fit['mu']).all()

def test_fit_pickling_cycle(tmp_path, monkeypatch):
    """
    Test that the model fit can be pickled and loaded into a new ParameterEstimator for cross-machine or collaborative use.
    """

    data_online = [np.array([1.2, 1.3, 1.6, 1.9])]
    data_offline = [np.array([1.8,1.9])]
    M = 1
    T = 2

    monkeypatch.chdir(tmp_path)
    est = ParameterEstimator(data_online, data_offline, M, T)
    est.fit = {
            'alpha' : np.ones((2,2,100)),
            'gamma' : np.ones((2,2,100)),
            'mu' : np.ones((1,2,100))
            }
    with open('fit.pkl', 'wb') as file:
        pickle.dump(est.fit, file)

    #Create a new estimator and load
    new_est = ParameterEstimator(data_online, data_offline, M, T)
    with open('fit.pkl', 'rb') as file:
        new_est.fit = pickle.load(file)

    assert (new_est.fit['alpha'] == est.fit['alpha']).all()
    assert (new_est.fit['gamma'] == est.fit['gamma']).all()
    assert (new_est.fit['mu'] == est.fit['mu']).all()


def test_class_accepts_various_data_input_types():
    """
    This function is to test that the estimator accepts various input types
    """
    #list of lists
    online = [[1,2,3], [4,5]]
    offline = [[2,2], [3,3]]
    est = ParameterEstimator(online, offline, M = 2, T = 5)
    assert isinstance(est.data_online, list)
    assert isinstance(est.data_offline, list)
    assert all(isinstance(x, np.ndarray) for x in est.data_online)
    assert all(isinstance(x, np.ndarray) for x in est.data_offline)
    assert np.array_equal(est.data_online[0], np.array([1,2,3]))
    assert np.array_equal(est.data_online[1], np.array([4,5]))
    assert np.array_equal(est.data_offline[0], np.array([2,2]))
    assert np.array_equal(est.data_offline[1], np.array([3,3]))

    # numpy array of the form np.array([[2,3],[3,4]])
    arr_online = np.array([[1,2,3],[4,5,6]])
    arr_offline = np.array([[7,8,9],[10,11,12]])
    est2 = ParameterEstimator(arr_online, arr_offline, M = 2, T = 12)
    assert isinstance(est2.data_online, list)
    assert isinstance(est2.data_offline, list)
    assert all(isinstance(x, np.ndarray) for x in est2.data_online)
    assert all(isinstance(x, np.ndarray) for x in est2.data_offline)
    assert np.array_equal(est2.data_online[0], np.array([1,2,3]))
    assert np.array_equal(est2.data_online[1], np.array([4,5,6]))
    assert np.array_equal(est2.data_offline[0], np.array([7,8,9]))
    assert np.array_equal(est2.data_offline[1], np.array([10,11,12]))
    
    #Already a list of numpy arrays. The class then should use the data as is
    list_of_arr_online = [np.array([1,2,3]), np.array([3,4,5])]
    list_of_arr_offline = [np.array([6,7,8]), np.array([10, 11, 12])]
    est3 = ParameterEstimator(list_of_arr_online, list_of_arr_offline, M = 2, T = 12)
    assert isinstance(est3.data_online, list)
    assert isinstance(est3.data_offline, list)
    assert all(isinstance(x, np.ndarray) for x in est3.data_online)
    assert all(isinstance(x, np.ndarray) for x in est3.data_offline)

    #Single array
    single_online = np.array([1,2])
    single_offline = np.array([4,5])
    est4 = ParameterEstimator(single_online, single_offline, M = 1, T = 5)
    assert isinstance(est4.data_online, list)
    assert isinstance(est4.data_offline, list)
    assert isinstance(est4.data_online[0], np.ndarray)
    assert isinstance(est4.data_offline[0], np.ndarray)
    assert np.array_equal(est4.data_online[0], np.array([1,2]))
    assert np.array_equal(est4.data_offline[0], np.array([4,5]))
