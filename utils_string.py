import numpy as np
from scipy.special import binom
from scipy.optimize import brentq


class TT_string(object):


    def __init__(self, L_string, N_tot, N_recruit):
        
        # Length of the binary string defining the clonotype
        self.L = L_string

        # Total number of clonotypes in the repertoire
        self.N_tot = N_tot

        # Number of recruited clonotypes (average)
        self.N_recr = N_recruit

        # Parameters of the expansion dynamics
        #self.TT_p = TT_params

        # Exponent of the recruitment probability
        self.gamma_recr = 1

        # Setting the parameter D0 of the recruitment prob to fix N_recruit
        self.D0 = brentq(lambda x : self._av_p_recr(x) - N_recruit/N_tot, 0.01, 5) 


    def gen_and_recruit_string(self, av_logtau=-4.0, std_logtau=2.5):
        """
        It generates the full repertoire and select the recruited clonotypes according to
        P_recruit = exp(-(H/D0)^gamma).
        The recruited string hamming distances are then translated into logtaus (affinitites)
        in such a way that their distribution has a given mean and standard deviation.
        """

        # Generating the strings and computing the hamming distance against the original antigen
        strings = np.random.randint(0, 2,size=(self.N_tot, self.L))
        h_dists = np.sum(strings, axis=1)

        # Selecting the clonotypes according to their recruitment prob
        recruit_probs = np.exp(-(h_dists/self.D0)**self.gamma_recr)
        unifs = np.random.rand(self.N_tot)

        self.recruit_strings = strings[unifs < recruit_probs,:]
        self.h_recruit = np.sum(self.recruit_strings, axis=1)
        self.logtaus = self.get_logtau(self.h_recruit, av_logtau, std_logtau)

        return self.recruit_strings, self.h_recruit, self.logtaus
    

    def get_logtau(self, hs, av_logtau=-4.0, std_logtau=2.5):
        """
        It transforms the hamming distnces in log-affinities by shifting their
        average and standard deviation to match the empirical ones
        """
        mean, std = self._mean_std_h_pass()
        b = std_logtau / std
        a = av_logtau + b*mean
        return a - b*np.array(hs, dtype=float)


    def get_pars(self):
        """
        Return the parameters as a dictionary
        """
        return {
            'L_string' : self.L, 
            'N_tot' : self.N_tot, 
            'N_recruit' : self.N_recr,
            'D0' : self.D0
        }
    

    def _av_p_recr(self, D0):
        """ 
        Average probability that a clonotype is recruited 
        """
        s = np.sum([binom(self.L, H) * np.exp(-(H/D0)**self.gamma_recr) for H in range(self.L+1)])
        return s*2**(-self.L)


    def _w_pass(self, H):
        """
        Probability weights (to be normalized) that a clonotype having hamming distance
        H is present among the recruted ones
        """
        return binom(self.L, H) * np.exp(-(H/self.D0)**self.gamma_recr)


    def _mean_std_h_pass(self):
        """
        Get the average and standard deviation of the recruited hamming distances
        """
        Hs = np.arange(self.L + 1)
        wh = self._w_pass(Hs)
        mean = np.sum(wh * Hs) / np.sum(wh)
        m2 = np.sum(wh * Hs**2) / np.sum(wh)
        return mean, np.sqrt(m2 - mean**2)
        


def comp_h_dist(strings, target_string):
    """
    Compute the hamming distance between a list of strings and
    a target string
    """
    return np.sum(np.array(strings) != np.array(target_string)[np.newaxis,:], axis=1)


def comp_coverage_h(strings, abunds, target_string):
    h_dists = comp_h_dist(strings, target_string)
    return abunds*h_dists