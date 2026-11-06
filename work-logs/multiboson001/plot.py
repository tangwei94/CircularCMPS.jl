"""Render the three left-canonical trajectories from one completed run directory."""
import csv
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

run = Path(sys.argv[1])
labels = {'noprep_precond': 'No preparation, preconditioned',
          'noprep_noprecond': 'No preparation, unpreconditioned',
          'prep100_precond': 'After 100 preparation steps, preconditioned'}
data = {}
for name in labels:
    path = run / f'{name}.tsv'
    if path.exists():
        with path.open() as f:
            data[name] = [{k: float(v) for k,v in r.items()} for r in csv.DictReader(f, delimiter='\t')]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
for name, rows in data.items():
    x = [r['iter'] for r in rows]
    axes[0].plot(x, [r['energy_paper'] for r in rows], label=labels[name])
    axes[1].semilogy(x, [r['cond_rhoR'] for r in rows], label=labels[name])
axes[0].set_ylabel('Energy density (manuscript units)')
axes[1].set_ylabel(r'$\mathrm{cond}_2(\rho_R)$')
for ax in axes:
    ax.set_xlabel('Left-canonical iteration')
    ax.grid(alpha=.2)
fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=1, frameon=False)
fig.tight_layout(rect=(0,.19,1,1))
fig.savefig(run/'trajectories.pdf')
fig.savefig(run/'trajectories.png', dpi=160)
