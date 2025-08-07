def main():

    import pandas as pd
    from o2o import BivariateHawkesProcessSimulator, ParameterEstimator, O2OAnalyzer

    M = int(input('Give the number of users:   '))
    T = int(input('Give the time window of activity (in days):   '))

    # Simulate data
    sim = BivariateHawkesProcessSimulator(M, T)
   #This avoids empty data
    while True:
        data_online, data_offline = sim.simulate()
        if all(len(o) > 0 for o in data_online) and all(len(f) > 0 for f in data_offline):
            break
    # Fit model
    print('Data generated!! Fitting started... This may take sometime \n')
    estimator = ParameterEstimator(data_online, data_offline, M, T)
    fit = estimator.fit_model()


    #Print estimation results
    base_estimation = estimator.base_and_CI()
    spillover, decay = estimator.spillover_and_decays_values_and_CI()
    percentages = estimator.spillover_percentage()

    print(base_estimation)
    print(' ')
    print(spillover)
    print(' ')
    print(decay)
    print(' ')
    print(percentages)
    print('')

    # Analyze
    analyzer = O2OAnalyzer(data_online, data_offline, M, T, fit)
    sizes = analyzer.sample_size()
    print(sizes)
    analyzer.plot_intensity()

if __name__ == "__main__":
    main()

