import numpy as np
import pickle



class BivariateHawkesProcessSimulator:
    """
    Simulator for a bivariate self-exciting point process (Hawkes process)
    with two marks (0: offline, 1: online) over a time window [0, T].
    """
    def __init__(self, M, T, alpha = None, gamma = None, mu = None):
        """
        Initialize simulator parameters.
        Args:
            - M (int): Number of independent sequences (e.g., users).
            - T (int): Length of observation window.
            - gamma (2x2 array, optional): Decay rates matrix.
            - alpha (2x2 array, optional): Reproduction matrix.
            - mu (Mx2 array, optional): Baseline rates for each user.
        """
        self.T = T
        self.M = M

        if gamma is  None:
            # Initialize decay rates gamma for each mark-to-mark excitation (if the user specifies it  will
            #overwrite the values wa use here)
            # gamma[i,j] is the decay rate of events of type j exciting type i
            self.gamma = np.array([[5., 1.], [3., 2.]])
        else:
            self.gamma = np.array(gamma)

        if alpha is  None:
            # Initialize the reproduction matrix alpha (if the user specifies it  will overwrite the values wa use here)
            # alpha[i,j] is the expected number of offspring of type i triggered from a parent of type
            self.alpha = np.array([[.6, .1], [.2, .8]])
        else:
            self.alpha = np.array(alpha)

        if mu is  None:
            #Initialize the baseline intensity mu (if the user specifies it  will overwrite the values wa use here)
            self.mu = np.zeros([M, 2])
            self.mu[:, 0] = .1 #Offline initial rate
            self.mu[:, 1] = .3 #Online initial rate
        else:
            self.mu = np.array(mu)

    def simulate(self, save_online_data = False, save_offline_data = False):
        """
        Run the simulation.

        Args:
            save_online_data (bool): Save online data to pickle if True.
            save_offline_data (bool): Save offline data to pickle if True.
            online_path (str): File path for online data pickle.
            offline_path (str): File path for offline data pickle.

        Returns:
            online_data (list of lists): Event times for mark 1 (online).
            offline_data (list of lists): Event times for mark 0 (offline).
        """
        np.random.seed(42)
        marks_outer = []
        N_list = []
        times_outer = []

        # Generate baseline (immigrant) events via Poisson((mu0 + mu1)*T)
        for m in range(self.M):
            N_list.append(np.random.poisson(np.sum(self.mu[m]) * self.T))
            times = []
            for i in range(N_list[m]):
                times.append(self.T * np.random.rand())
            times_outer.append(times)


        #Assign marks to baseline events
        for m in range(self.M):
            marks = []
            for i in range(N_list[m]):
                #Probability of mark 0 is mu0/(mu0+mu1)
                if (self.mu[m,0]/(self.mu[m,0] + self.mu[m,1])) > np.random.rand():
                    marks.append(0)
                else:
                    marks.append(1)
            marks_outer.append(marks)

        #Generate offspring via branching until no unprocessed events remain
        cnt = 0
        for m in range(self.M):
            while cnt < len(times_outer[m]):
                for i in range(cnt, len(times_outer[m])):
                    k = marks_outer[m][i]
                    cnt+=1

                    g = np.random.poisson(self.alpha[0,k])
                    for j in range(g):
                        tnew = times_outer[m][i] + np.random.exponential(1. / self.gamma[0,k])
                        if tnew < self.T:
                            times_outer[m].append(tnew)
                            marks_outer[m].append(0)
                            N_list[m]+=1

                    g = np.random.poisson(self.alpha[1,k])
                    for j in range(g):
                        tnew = times_outer[m][i] + np.random.exponential(1. / self.gamma[1,k])
                        if tnew < self.T:
                            times_outer[m].append(tnew)
                            marks_outer[m].append(1)
                            N_list[m]+=1

        #Sort each sequence's events by time, keeping marks in sync
        for m in range(self.M):
            if len(times_outer[m]) > 0:
                times_outer[m], marks_outer[m] = (list(t) for t in zip(*sorted(zip(times_outer[m], marks_outer[m]))))
            else:
                times_outer[m], marks_outer[m] = [], []

        #Separate times by mark
        offline_data = [] #times for mark 0
        for m in range(len(times_outer)):
            times0 = []
            for i in range(len(times_outer[m])):
                if marks_outer[m][i] == 0:
                    times0.append(times_outer[m][i])
            offline_data.append(times0)


        online_data = [] #times for mark 1
        for m in range(len(times_outer)):
            times1 = []
            for i in range(len(times_outer[m])):
                if marks_outer[m][i] == 1:
                    times1.append(times_outer[m][i])
            online_data.append(times1)

        #optional pickling to save the online and offline data
        if save_online_data:
            with open('online_data.pkl', 'wb') as file:
                pickle.dump(online_data, file)

        if save_offline_data:
            with open('offline_data.pkl', 'wb') as file:
                pickle.dump(offline_data, file)

        return online_data, offline_data
