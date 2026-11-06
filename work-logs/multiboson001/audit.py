"""Audit saved trajectories and generate summary including iteration zero."""
import csv
import math
import sys
from pathlib import Path

run = Path(sys.argv[1])
names = ['noprep_precond','noprep_noprecond','prep100_precond']
data = {}
for name in names:
    with (run/f'{name}.tsv').open() as f:
        rows = [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f,delimiter='\t')]
    assert [r['iter'] for r in rows] == list(range(len(rows)))
    assert all(all(math.isfinite(v) for v in r.values()) for r in rows)
    assert all(r['energy_paper'] == r['energy_raw']/2 for r in rows)
    data[name] = rows
ref = data['prep100_precond'][-1]['energy_paper']
fields = ['trajectory','iterations','final_energy_paper','final_gnorm','initial_cond','final_cond','max_cond','min_cond','min_cond_iteration','max_gauge_residual','max_hermiticity','min_eigenvalue','converged']
with (run/'summary.tsv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=fields,delimiter='\t'); writer.writeheader()
    for name,rows in data.items():
        first,last=rows[0],rows[-1]
        low=min(rows,key=lambda r:r['cond_rhoR'])
        writer.writerow(dict(trajectory=name,iterations=int(last['iter']),final_energy_paper=last['energy_paper'],final_gnorm=last['gnorm'],initial_cond=first['cond_rhoR'],final_cond=last['cond_rhoR'],max_cond=max(r['cond_rhoR'] for r in rows),min_cond=low['cond_rhoR'],min_cond_iteration=int(low['iter']),max_gauge_residual=max(r['gauge_residual'] for r in rows),max_hermiticity=max(r['hermiticity'] for r in rows),min_eigenvalue=min(r['eigen_min'] for r in rows),converged=last['gnorm']<=1e-6))
with (run/'energy-errors.tsv').open('w') as f:
    f.write('trajectory\titer\tenergy_reference_paper\tabs_energy_difference_paper\n')
    for name,rows in data.items():
        for r in rows:
            f.write(f"{name}\t{int(r['iter'])}\t{ref}\t{abs(r['energy_paper']-ref)}\n")
print('PASS: contiguous iteration indices, finite values, energy conversion; summaries include initial row.')
