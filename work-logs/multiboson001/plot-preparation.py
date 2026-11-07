"""Validate and plot the diagonal-gauge preparation diagnostic."""
import csv
import math
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

run=Path(sys.argv[1])
with (run/'preparation.tsv').open() as f:
    rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f,delimiter='\t')]
assert len(rows)==101
assert [r['iter'] for r in rows]==list(range(101))
assert all(all(math.isfinite(v) for v in r.values()) for r in rows)
assert all(r['energy_paper']==r['energy_raw']/2 for r in rows)
# Gauge conversion must preserve energy to numerical precision.
assert max(r['energy_conversion_abs_error_paper'] for r in rows)<1e-7
summary={
'initial_energy_paper':rows[0]['energy_paper'],
'final_energy_paper':rows[-1]['energy_paper'],
'initial_cond_rhoR':rows[0]['cond_rhoR'],
'final_cond_rhoR':rows[-1]['cond_rhoR'],
'min_cond_rhoR':min(r['cond_rhoR'] for r in rows),
'max_cond_rhoR':max(r['cond_rhoR'] for r in rows),
'final_gnorm_diagonal':rows[-1]['gnorm_diagonal'],
'max_energy_conversion_abs_error_paper':max(r['energy_conversion_abs_error_paper'] for r in rows),
'max_gauge_residual':max(r['gauge_residual'] for r in rows),
'max_hermiticity':max(r['hermiticity'] for r in rows),
'min_eigenvalue':min(r['eigen_min'] for r in rows),
}
with (run/'summary.tsv').open('w') as f:
    f.write('quantity\tvalue\n')
    for k,v in summary.items(): f.write(f'{k}\t{v}\n')
fig,ax=plt.subplots(1,2,figsize=(9,3.3))
x=[r['iter'] for r in rows]
ax[0].plot(x,[r['energy_paper'] for r in rows])
ax[0].set_ylabel('Energy density (manuscript units)')
ax[1].semilogy(x,[r['cond_rhoR'] for r in rows])
ax[1].set_ylabel(r'$\mathrm{cond}_2(\rho_R)$ after left canonicalization')
for a in ax:
    a.set_xlabel('Diagonal-gauge preparation iteration')
    a.grid(alpha=.2)
fig.tight_layout()
fig.savefig(run/'preparation.pdf')
fig.savefig(run/'preparation.png',dpi=160)
for k,v in summary.items(): print(k,v)
print('PASS: 101 contiguous, finite rows; energy conversion invariant.')
