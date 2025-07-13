import numpy as np
import pandas as pd
import pickle
#import stan

try:
    import stan
except ImportError:
    import pystan as stan

# Allow running Stan in notebooks without loop errors
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

class ParameterEstimator:
    """
    This class estimates the model parameters and outputs tables of the spillover estimation, decay rate estimation,
    and the backround rates as well as their corresponding 95% confidence intervals
    """
    def __init__(self, data_online, data_offline, M, T):

        """
        Args:
            data_online (list of np.arrays): Timestamps of online events per user
            data_offline (list of np.arrays): Timestamps of offline events per user
            M (int): Number of users
            T (int): Total observation time
        """
        
        self.data_online = self._ensure_list_of_arrays(data_online)
        self.data_offline = self._ensure_list_of_arrays(data_offline)
        self.M = M
        self.T = T
        self.fit = None  #stores the posterior samples after fitting

        if len(self.data_online) != M or len(self.data_offline) != M:
            raise ValueError("Length of online and offline data should match M")

    @staticmethod
    def _ensure_list_of_arrays(data):
        """
        This static function ensures that the data is a list of numpy arrays if not already
        """
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                #If a single array, wrap in a list
                return [np.array(data)]
            elif data.ndim == 2:
                return [np.array(row) for row in data]
        elif isinstance(data, list):
            return [np.array(row) for row in data]


    #Define the first method in this class: The parameter estimation using Stan
    def fit_model(self, data_online, data_offline, M, T, save_fit = False):
        """
        Estimate parameters using Stan.

        Args:
            data_online (list): Online events
            data_offline (list): Offline events
            M (int): Number of users
            T (int): Time period
            save_fit (bool): Whether to save the Stan fit to a pickle file

        Returns:
            fit: Posterior samples
        """
        np.random.seed(42)

        na_online = [len(i) for i in self.data_online ] #online sizes
        nb_offline = [len(i) for i in self.data_offline ] #offline sizes

        #pad online data to feed it to Stan (Stan can't process ragged data)
        online_padded_data = []
        for i in data_online:
            online_padded_data.append(np.pad(i, (0, max(na_online)- len(i)), constant_values = -1))

        #pad offline data to feed it to Stan (Stan can't process ragged data)
        offline_padded_data = []
        for i in data_offline:
            offline_padded_data.append(np.pad(i, (0, max(nb_offline)- len(i)), constant_values = -1))

        #Stan code
        model_code = """
        data {
          int<lower=1> M;                  // number of gangs
          array[M] int<lower=1> Na;        // list of lenght for each on_data
          array[M] int<lower=1> Nb;        // list of lenght for each off_data
          int<lower=1> maxNa;              // maximum over Na
          int<lower=1> maxNb;              // maximum over Nb
          array[M] vector[maxNa] ta;
          array[M] vector[maxNb] tb;
          int<lower=0> T;
        }
        parameters {
          matrix<lower=0>[M,2] mu;                     // baseline
          matrix<lower=0>[2,2] gamma;                  // decay
          matrix<lower=0, upper=1>[2,2] alpha;         // adjacency
        }
        transformed parameters {
          array[M] vector[maxNa] lam_a;
          array[M] vector[maxNb] lam_b;

          // initialize first elements
          for (m in 1:M) {
            lam_a[m][1] = mu[m,1];
            lam_b[m][1] = mu[m,2];
          }

          // lam online
          for (m in 1:M) {
            for (j in 1:Na[m]) {
              lam_a[m][j] = mu[m,1];
              for (k in 1:(j-1)) {
                if (ta[m][j] > ta[m][k] && ta[m][j] != -1 && ta[m][k] != -1) {
                  lam_a[m][j] += alpha[1,1] * gamma[1,1] * exp(-gamma[1,1] * (ta[m][j] - ta[m][k]));
                }
              }
              for (k in 1:Nb[m]) {
                if (ta[m][j] > tb[m][k] && ta[m][j] != -1 && tb[m][k] != -1) {
                  lam_a[m][j] += alpha[1,2] * gamma[1,2] * exp(-gamma[1,2] * (ta[m][j] - tb[m][k]));
                }
              }
            }

            // lam offline
            for (j in 1:Nb[m]) {
              lam_b[m][j] = mu[m,2];
              for (k in 1:(j-1)) {
                if (tb[m][j] > tb[m][k] && tb[m][j] != -1 && tb[m][k] != -1) {
                  lam_b[m][j] += alpha[2,2] * gamma[2,2] * exp(-gamma[2,2] * (tb[m][j] - tb[m][k]));
                }
              }
              for (k in 1:Na[m]) {
                if (tb[m][j] > ta[m][k] && tb[m][j] != -1 && ta[m][k] != -1) {
                  lam_b[m][j] += alpha[2,1] * gamma[2,1] * exp(-gamma[2,1] * (tb[m][j] - ta[m][k]));
                }
              }
            }
          }
        }

        model {
          // priors
          alpha[1,1] ~ beta(1,1);
          alpha[1,2] ~ beta(1,1);
          alpha[2,1] ~ beta(1,1);
          alpha[2,2] ~ beta(1,1);

          for (m in 1:M) {
            mu[m,1] ~ cauchy(0,5);
            mu[m,2] ~ cauchy(0,5);
          }
          gamma[1,1] ~ cauchy(0,5);
          gamma[2,1] ~ cauchy(0,5);
          gamma[1,2] ~ cauchy(0,5);
          gamma[2,2] ~ cauchy(0,5);

          // likelihood maximization using the Shoenberg approximation
          for (m in 1:M) {
            for (j in 1:Na[m]) {
              target += log(lam_a[m][j]);
            }
            for (j in 1:Nb[m]) {
              target += log(lam_b[m][j]);
            }
            target += -mu[m,1] * T -mu[m,2] * T - (alpha[1,1] + alpha[2,1]) * Na[m] - (alpha[1,2] + alpha[2,2]) * Nb[m];
          }
        }
        """

        #it will take some time
        hawkes_data = {"Na": na_online, "Nb":nb_offline, "ta":np.array(online_padded_data)
                       , "tb": np.array(offline_padded_data), "maxNa" :max(na_online)
                       , "maxNb": max(nb_offline), "M": self.M, "T": self.T}
        posterior = stan.build(model_code, data = hawkes_data, random_seed=2)
        fit = posterior.sample(num_chains=1, num_samples=1000)

        #optinal saving for the fit file
        if save_fit:
            with open('fit.pkl', 'wb') as file:
                pickle.dump(fit, file)
        self.fit = fit
        return fit

    def spillover_and_decays_values_and_CI(self, save_alpha = False, save_gamma = False):
        """
    	This method constructs the tables of the O2O spillover as well as the O2O decay rate and their conrresponding 95% CI.
    	In our convention, the marked 0 events correspond to online, and the marked 1 events correspond to offline.
        - alpha_{00}: 0 --> 0 online to online spillover
        - alpha_{11}: 1 --> 1 offline to offline spillover
        - alpha_{01}: 1 --> 0 offline to online spillover
        - alpha_{10}: 0 --> 1 online to offline spillover

        - gamma_{00}: 0 --> 0 online to online decay rate
        - gamma_{11}: 1 --> 1 offline to offline decay rate
        - gamma_{01}: 1 --> 0 offline to online decay rate
        - gamma_{10}: 0 --> 1 online to offline decay rate

	Args:
            - save_alpha, save_beta: are declared when wanting to save the alpha and gamma tables as csv files
        Returns: 
            - pd.DataFrame of alpha and gamma and their 95% CI
        """
        fit = self.fit

        #Spillover
        alpha_on_to_on =  '{:.3f}'.format(np.mean(fit['alpha'][0,0], axis=0))
        alpha_off_to_off = '{:.3f}'.format(np.mean(fit['alpha'][1,1], axis=0))
        alpha_on_to_off = '{:.3f}'.format(np.mean(fit['alpha'][1,0], axis=0))
        alpha_off_to_on = '{:.3f}'.format(np.mean(fit['alpha'][0,1], axis=0))

        #Decay rate
        gamma_on_to_on =  '{:.3f}'.format(np.mean(fit['gamma'][0,0], axis=0))
        gamma_off_to_off = '{:.3f}'.format(np.mean(fit['gamma'][1,1], axis=0))
        gamma_on_to_off = '{:.3f}'.format(np.mean(fit['gamma'][1,0], axis=0))
        gamma_off_to_on = '{:.3f}'.format(np.mean(fit['gamma'][0,1], axis=0))

        #Spillover 95% CI
        alpha_on_to_on_CI = ['{:.3f}'.format(np.percentile(fit['alpha'][0,0], 5, axis=0))
                             , '{:.3f}'.format(np.percentile(fit['alpha'][0,0], 95, axis=0))]
        alpha_off_to_off_CI =  ['{:.3f}'.format(np.percentile(fit['alpha'][1,1], 5, axis=0))
                             , '{:.3f}'.format(np.percentile(fit['alpha'][1,1], 95, axis=0))]
        alpha_on_to_off_CI =  ['{:.3f}'.format(np.percentile(fit['alpha'][1,0], 5, axis=0))
                             , '{:.3f}'.format(np.percentile(fit['alpha'][1,0], 95, axis=0))]
        alpha_off_to_on_CI =  ['{:.3f}'.format(np.percentile(fit['alpha'][0,1], 5, axis=0))
                             , '{:.3f}'.format(np.percentile(fit['alpha'][0,1], 95, axis=0))]
        #Decay rate 95% CI
        gamma_on_to_on_CI = ['{:.3f}'.format(np.percentile(fit['gamma'][0,0], 5, axis=0))
                             , '{:.3f}'.format(np.percentile(fit['gamma'][0,0], 95, axis=0))]
        gamma_off_to_off_CI = ['{:.3f}'.format(np.percentile(fit['gamma'][1,1], 5, axis=0))
                             , '{:.3f}'.format(np.percentile(fit['gamma'][1,1], 95, axis=0))]
        gamma_on_to_off_CI = ['{:.3f}'.format(np.percentile(fit['gamma'][1,0], 5, axis=0))
                             , '{:.3f}'.format(np.percentile(fit['gamma'][1,0], 95, axis=0))]
        gamma_off_to_on_CI = ['{:.3f}'.format(np.percentile(fit['gamma'][0,1], 5, axis=0))
                             , '{:.3f}'.format(np.percentile(fit['gamma'][0,1], 95, axis=0))]
        #Table of spillover effect and its 95% CI
        alpha_data = [
            ['Online to Online Spillover', alpha_on_to_on, alpha_on_to_on_CI],
            ['Offline to Offline Spillover', alpha_off_to_off, alpha_off_to_off_CI],
            ['Online to Offline Spillover', alpha_on_to_off, alpha_on_to_off_CI],
            ['Offline to Online Spillover', alpha_off_to_on, alpha_off_to_on_CI]
        ]
        alpha_df = pd.DataFrame(alpha_data, columns=['Effect', 'Value', '95% CI'])

        #Table of decay rate and its 95% CI
        gamma_data = [
            ['Online to Online Decay', gamma_on_to_on, gamma_on_to_on_CI],
            ['Offline to Offline Decay', gamma_off_to_off, gamma_off_to_off_CI],
            ['Online to Offline Decay', gamma_on_to_off, gamma_on_to_off_CI],
            ['Offline to Online Decay', gamma_off_to_on, gamma_off_to_on_CI]
        ]
        gamma_df = pd.DataFrame(gamma_data, columns=['Effect', 'Value', '95% CI'])

        if save_alpha:
            alpha_df.to_csv('alpha_estimates.csv', index = False)
        if save_gamma:
            gamma_df.to_csv('gamma_estimates.csv', index = False)

        return pd.DataFrame(alpha_df), pd.DataFrame(gamma_df)


    def base_and_CI(self, save_mu = False):
        """
        Online and offline baseline intensities and their 95% for each user    
            Args:
                save_mu (bool): Save the output table to 'estimated_mu.csv'

            Returns:
                pd.DataFrame of mu and 95%CI for each user online and offline
        """
        fit = self.fit

        M = len(fit['mu']) #Number of users
        labels = [f'User {i}' for i in range(1, M + 1)] #Generate labels as User1, User2...

        baseline = np.mean(fit['mu'], axis = 2)
        def base(user, x):
            return '{:.3f}'.format(baseline[labels.index(user)][x])
        def base_CI(user, x):
            return  ['{:.3f}'.format(np.percentile(fit['mu'][labels.index(user),x], 5))
                     , '{:.3f}'.format(np.percentile(fit['mu'][labels.index(user),x], 95))]

        table_base = {'User': labels,

            'Online baseline intensity' :[base(i, 0) for i in labels],

            'Online baseline intensity 95% CI' :[base_CI(i, 0) for i in labels],

            'Offline baseline intensity' :[base(i, 1) for i in labels],

            'Offline baseline intensity 95% CI' :[base_CI(i, 1) for i in labels]
            }
        df = pd.DataFrame(table_base)
        if save_mu:
            df.to_csv('estimated_mu.csv', index = False)
        return pd.DataFrame(table_base)

    def spillover_percentage(self, save_percentages = False):
        """
        Computes what percent of events of one type are caused by the other (spillover percentages).

        Returns:
            pd.DataFrame: Table showing %Online→Offline and %Offline→Online per user and in aggregate
        """

        fit = self.fit
        M = len(fit['mu']) #Number of users

        labels = [f'User {i}' for i in range(1, M + 1)] # generate labels
        baseline = np.mean(fit['mu'], axis = 2)
        adj = np.mean(fit['alpha'],2)

        #Baseline intensity for each user
        def base(user, x):
            return float('{:.3f}'.format(baseline[labels.index(user)][x]))

        # n_{online}: the total expected number of online events
        def n_on_total(user):
            n_on = ((-1+adj[1,1])*base(user, 0) - adj[0,1]*base(user, 1)) / (adj[0,0] + adj[0,1]*adj[1,0] + adj[1,1] - adj[0,0]*adj[1,1]-1)
            return n_on


        # n_{offline}: the total expected number of offline events
        def n_off_total(user):
            n_off = ((-1+adj[0,0])*base(user, 1) - adj[1,0]*base(user, 0)) / (adj[0,0] + adj[0,1]*adj[1,0] + adj[1,1]- adj[0,0]*adj[1,1]-1)
            return n_off

        # n0_{Online}: the expected number of Online events when alpha_{01} = 0 (e.g. there is no contribution of offline events)
        def n_online0(user):
            n_online = (((-1+adj[1,1])*base(user, 0)) / (adj[0,0] + adj[1,1]- adj[0,0]*adj[1,1]-1))
            return n_online

        #n0_{Offline}, the expected number of Offline events when alpha_{10} = 0 (e.g. there is no contribution of online events)
        def n_offline0(user):
            n_offline = ((-1+adj[0,0])*base(user, 1)) / (adj[0,0]  + adj[1,1]- adj[0,0]*adj[1,1]-1)
            return n_offline


        # the percentage %Online -> Offline = 100*(1-n0_offline/n_offline) (see eq 18)
        def offline_to_online_percent(user):
            percent = n_offline0(user) / n_off_total(user)

            return '{:.3f}'.format((1 - percent)*100)

        # the percentage %Offline -> Online = 100*(1-n0_online/n_online)
        def online_to_offline_percent(user):
            percent =  n_online0(user) / n_on_total(user)
            return '{:.3f}'.format((1 - percent)*100)


        #calculate the Offline -> Online aggregate
        offline0=[]
        offlinetot = []
        for user in labels:
            offline0.append(n_offline0(user))
            offlinetot.append(n_off_total(user))
        offline_to_online_aggr = '{:.3f}'.format(100*(1-(np.sum(offline0)/np.sum(offlinetot))))

        #calculate the Online -> Offline aggregate
        online0=[]
        onlinetot = []
        for user in labels:
            online0.append(n_online0(user))
            onlinetot.append(n_on_total(user))
        online_to_offline_aggr = '{:.3f}'.format(100*(1-(np.sum(online0) / np.sum(onlinetot))))

        percent_table = {'User': [*labels, 'Aggregate'],
        '% Online  → Offline' : [*[online_to_offline_percent(i) for i in labels], online_to_offline_aggr],
        '% Offline → Online' : [*[offline_to_online_percent(i) for i in labels], offline_to_online_aggr]}
        df = pd.DataFrame(percent_table)
        if save_percentages:
            df.to_csv('estimated_percentages_effects.csv', index = False)

        return pd.DataFrame(percent_table)





        
        
