import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution

# Data Loading
mrna_df = pd.read_csv("mrna.txt", delim_whitespace=True)
protein_df = pd.read_csv("protein.txt", delim_whitespace=True)

t_obs = mrna_df.iloc[:, 0].values
mrna_obs = mrna_df.iloc[:, 1:6].values
protein_obs = protein_df.iloc[:, 1:6].values
obs_data = np.hstack([mrna_obs, protein_obs])

y0_mrna = mrna_df.iloc[0, 1:6].values
y0_prot = protein_df.iloc[0, 1:6].values
y0 = np.concatenate([y0_mrna, y0_prot])

# ODE model builder
def make_syst(params):
    def syst(t, yy):
        y = np.maximum(yy, 0)
        foo_m, bar_m, sqk_m, drp_m, wbl_m = y[:5]
        foo, bar, sqk, drp, wbl = y[5:]

        Dm = 3.0
        Df, Db, Ds, Dd, Dw = params[0:5]
        Vf, Vb, Vs, Vd, Vw = params[5:10]
        kfb, kfd, kbf, kbd, ksd, kdw, kds, kwb = params[10:18]
        nfb, nfd, nbf, nbd, nsd, ndw, nds, nwb = params[18:26]
        Vprot = 1.0

        qfb = (bar / kfb) ** nfb
        qfd = (drp / kfd) ** nfd
        dfoo_m = Vf * (qfb + qfd) / (1 + qfb + qfd) - Dm * foo_m

        qbf = (foo / kbf) ** nbf
        qbd = (drp / kbd) ** nbd
        dbar_m = Vb * qbf / (1 + qbf) / (1 + qbd) - Dm * bar_m

        qsd = (drp / ksd) ** nsd
        dsqk_m = Vs / (1 + qsd) - Dm * sqk_m

        qdw = (wbl / kdw) ** ndw
        qds = (sqk / kds) ** nds
        ddrp_m = Vd / (1 + qdw) / (1 + qds) - Dm * drp_m

        qwb = (bar / kwb) ** nwb
        dwbl_m = Vw * qwb / (1 + qwb) - Dm * wbl_m

        dfoo_p = Vprot * foo_m - Df * foo
        dbar_p = Vprot * bar_m - Db * bar
        dsqk_p = Vprot * sqk_m - Ds * sqk
        ddrp_p = Vprot * drp_m - Dd * drp
        dwbl_p = Vprot * wbl_m - Dw * wbl

        return np.array([
            dfoo_m, dbar_m, dsqk_m, ddrp_m, dwbl_m,
            dfoo_p, dbar_p, dsqk_p, ddrp_p, dwbl_p
        ])
    return syst

# Define function to simulate with best-fit parameters
def simulate_model(params):
    syst_fn = make_syst(params)
    sol = solve_ivp(syst_fn, (t_obs[0], t_obs[-1]), y0, t_eval=t_obs, method='RK45', max_step=1.0)
    return sol.t, sol.y.T if sol.success else None


best_params = params_6 = np.array([
 1.05220710e-01, 41.4981184, 67.8279516, 0.05316894, 0.01, 100.0,
 41.6048422, 40.1652230, 46.3122811, 6.83717255, 68.9484631, 100.0,
 37.9875317, 49.8197619, 100.0, 24.8332881, 45.3002651, 62.7585824,
 4.79807388, 5.0, 2.45204944, 1.00000015, 1.0, 5.0, 2.53747629, 3.86971758
])




t_sim, sim_data = simulate_model(best_params)

labels = ['foo_m', 'bar_m', 'sqk_m', 'drp_m', 'wbl_m', 'foo', 'bar', 'sqk', 'drp', 'wbl']


fig, axes = plt.subplots(5, 2, figsize=(12, 10))
axes = axes.flatten()
for i in range(10):
    ax = axes[i]
    ax.plot(t_obs, obs_data[:, i], 'o', label='Observed')
    ax.plot(t_sim, sim_data[:, i], '-', label='Simulated')
    ax.set_title(labels[i])
    ax.set_xlabel('Time')
    ax.set_ylabel('Concentration')
    ax.legend()

plt.tight_layout()

plt.savefig("simulation_vs_observed.png", dpi=300) 
print("saved as simulation_vs_observed.png")

plt.show()
