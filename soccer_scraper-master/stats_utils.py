import numpy as np
import scipy.stats as st

def get_stats(a25, a75):
    return ((a25 + a75)/2, (a75 - m)/st.norm.ppf(.75))


def pdf(x, m, s):
    num = np.e**((-1*(x-m)**2)/(2 * (s**2)))
    denom = s * (2*(np.pi**.5))
    return num/denom

