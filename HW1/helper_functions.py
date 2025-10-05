import pandas as pd
import numpy as np

def bass_model(t, p, q, M):
    N = np.zeros(len(t))
    S = np.zeros(len(t))
    for i in range(len(t)):
        S[i] = p*(M-N[i-1]) + (q/M)*N[i-1]*(M-N[i-1]) if i>0 else p*M
        N[i] = N[i-1] + S[i] if i>0 else S[i]
    return S

# --- Simulate Bass model with estimated parameters ---
def bass_model_sim(p, q, M, n_periods):
    N = np.zeros(n_periods)
    S = np.zeros(n_periods)
    for t in range(n_periods):
        S[t] = p*(M - N[t-1]) + (q/M)*N[t-1]*(M - N[t-1]) if t>0 else p*M
        N[t] = N[t-1] + S[t] if t>0 else S[t]
    return S, N

