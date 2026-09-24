import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('results/full_cycle')
rows=list(csv.DictReader((p/'monitor_gls.csv').open()))
def a(k): return np.array([float(r[k]) for r in rows])
t=a('time_s')
fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
axs[0,0].plot(t,a('J_min'),label='minimum J');axs[0,0].plot(t,a('J_max'),label='maximum J');axs[0,0].axhline(1,color='gray',ls='--');axs[0,0].set_ylabel('Local volume ratio J');axs[0,0].legend()
axs[0,1].plot(t,a('J_minus_1_RMS'));axs[0,1].set_ylabel('Volume-weighted RMS(J - 1)')
axs[1,0].plot(t,a('J_mean'));axs[1,0].axhline(1,color='gray',ls='--');axs[1,0].set_ylabel('Mean myocardial J');axs[1,0].ticklabel_format(useOffset=False)
axs[1,1].plot(t,a('V_mL'));axs[1,1].set_ylabel('Estimated cavity volume (mL)')
for ax in axs.flat: ax.set_xlabel('Time (s)');ax.grid(alpha=.25)
fig.suptitle('PDF HO GLS: 200 steps converged; local incompressibility remains poor',fontsize=13)
fig.savefig(p/'summary.png',dpi=160)
