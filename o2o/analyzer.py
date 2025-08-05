import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




class O2OAnalyzer:
    """
    This class provides analysis tools for online and offline event data. It includes:
        - A summary table showing the number of online activities (e.g., social media posts) and offline activities
          (e.g., incidents such as shootings, assaults, etc.) per user.
        - Plots of the coupled conditional intensities for online and offline events over time, for each user.
    """
    def __init__(self, data_online, data_offline, M, T, fit):
        """
        Initializes the O2OAnalyzer.

        Args:
            data_online (list of arrays): Timestamps of online events per user.
            data_offline (list of arrays): Timestamps of offline events per user.
            M (int): Number of users.
            T (int): Observation period (in days).
            fit (dict): Posterior fit object containing sampled parameters (mu, alpha, gamma).
        """
        self.data_online = data_online
        self.data_offline = data_offline
        self.M = M
        self.T = T
        self.fit = fit

    def sample_size(self, save_sizes = False):
        """
        Generates a table with the number of online and offline events per user.

        Args:
            save_sizes (bool): If True, saves the table as 'sizes.csv'.

        Returns:
            pd.DataFrame: Table of online and offline event counts per user.
        """

        M = self.M
        data_online = self.data_online
        data_offline = self.data_offline
        #generate a list of labels (we label each user by User1, User2...)
        labels = [f'User {i}' for i in range(1, M + 1)]

        na_online = [len(i) for i in data_online ] #online sizes for each user
        nb_offline = [len(i) for i in data_offline ] #offline sizes for each user

        #create the table that displays the sizes of activities online and offline for each user
        table_sizes = {'User': labels,
        'Online sample size':  na_online,
        'Offline sample size' :  nb_offline}
        sizes = pd.DataFrame(table_sizes)

        #optional: save the table to a csv file
        if save_sizes:
            sizes.to_csv('sizes.csv', index=None)

        return sizes

    def plot_intensity(self, save_figs = False):
        """
        Plots the coupled conditional intensities (online and offline) over time for each user.

        Args:
            save_figs (bool): If True, saves each user's plot as an EPS file.

        Returns:
            None
        """

        M = self.M
        fit = self.fit
        T = self.T
        data_online = self.data_online
        data_offline = self.data_offline

        base_list = np.mean(fit['mu'], axis = 0) #baseline intensity list (for each user)
        adj = np.mean(fit['alpha'], axis = 0) # the alpha matrix
        decay = np.mean(fit['gamma'], axis = 0)    # the decay matrix
        labels = [f'User {i}' for i in range(1, M + 1)] #generate labels

        times1 = data_online
        times2 = data_offline
        t = np.arange(0.1, T, 0.1)

        intensity_on_list = []
        intensity_off_list = []

       # intensity_on = np.zeros(len(t))
       # intensity_off = np.zeros(len(t))

        #the conditional intensity for the online events
        for k in range(len(data_online)):
            intensity_on = np.zeros(len(t))
            for i in range(len(t)):
                intensity_on[i] = base_list[k][0]
                for j in range(len(times1[k])):
                    if t[i] > times1[k][j]:
                        if t[i] == times1[k][j]:
                            j+=1 #avoid vinishing time differences
                        intensity_on[i] += adj[0,0] * decay[0,0]* np.exp(-decay[0,0] * (t[i] - times1[k][j]))
                for j in range(len(times2[k])):
                    if t[i] > times2[k][j]:
                        if t[i] == times2[k][j]:
                            j+=1 #avoid vanishing time differences
                        intensity_on[i] += adj[0,1]*decay[0,1]*np.exp(-decay[0,1]*(t[i]-times2[k][j]))
            intensity_on_list.append(intensity_on)


        #the conditional intensity for the offline events
        for k in range(len(data_offline)):
            intensity_off = np.zeros(len(t))
            for i in range(len(t)):
                intensity_off[i] = base_list[k][1]
                for j in range(len(times2[k])):
                    if t[i] > times2[k][j]:
                        if t[i] == times2[k][j]:
                            j+=1 #avoid vinishing time differences
                        intensity_off[i] += adj[1,1] * decay[1,1]* np.exp(-decay[1,1] * (t[i] - times2[k][j]))
                for j in range(len(times1[k])):
                    if t[i] > times1[k][j]:
                        if t[i] == times1[k][j]:
                            j+=1 #avoid vinishing time differences
                        intensity_off[i] += adj[1,0]*decay[1,0]*np.exp(-decay[1,0]*(t[i]-times1[k][j]))
            intensity_off_list.append(intensity_off)

        #plotting
        for i in range(M):
            fig, ax = plt.subplots(figsize = (16,5))
            ax2 = plt.twinx()
            ax.plot(t, intensity_on_list[i], color = 'r', label = 'Online Activity') #offline intensity
            ax2.plot(t, intensity_off_list[i], color = 'black',  label = 'Offline Activity') #online intensity
            #ax.set_xlabel('Time', fontsize=30)
            ax.set_ylabel('Offline Intensity', fontsize=25)
            ax2.set_ylabel('Online Intensity', fontsize=25)
            #ax.set_yticklabels(ax.get_yticks(), fontsize=12)
            ax.tick_params(axis='x', labelsize=25)
            ax.tick_params(axis='y', labelsize=25)
            ax2.tick_params(axis='y', labelsize=25)
            ax.legend(loc = 'upper left', fontsize=25)
            ax2.legend(loc = 'upper right', fontsize=25)
            plt.title(labels[i], fontsize=25)
            ax.set_xlabel('Time (in days)', fontsize = 25)
            if save_figs:
                plt.savefig(f"{labels[i]}_intensity.eps")
        plt.show()
