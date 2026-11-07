"""Compare the first 100 iterations using existing recorded trajectories."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = Path(__file__).resolve().parent
out = base / 'results' / 'first100-comparison'
out.mkdir(exist_ok=True)
sources = [
    ('Diagonal gauge', '20261107-134511-preparation/preparation.tsv', '#009E73', '-'),
    ('Left canonical, with preconditioner', '20261106-202812-seed1/noprep_precond.tsv', '#0072B2', '-'),
    ('Left canonical, without preconditioner', '20261106-202812-seed1/noprep_noprecond.tsv', '#D55E00', '--'),
]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
with (out / 'curves.tsv').open('w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['trajectory', 'source', 'iter', 'energy_paper', 'cond_rhoR'])
    for label, source, color, style in sources:
        with (base / 'results' / source).open() as stream:
            rows = [r for r in csv.DictReader(stream, delimiter='\t') if int(r['iter']) <= 100]
        assert [int(r['iter']) for r in rows] == list(range(101))
        x = [int(r['iter']) for r in rows]
        axes[0].plot(x, [float(r['energy_paper']) for r in rows], color=color, linestyle=style, lw=1.8, label=label)
        axes[1].semilogy(x, [float(r['cond_rhoR']) for r in rows], color=color, linestyle=style, lw=1.8, label=label)
        for r in rows:
            writer.writerow([label, source, r['iter'], r['energy_paper'], r['cond_rhoR']])
axes[0].set_ylabel('Energy density (manuscript units)')
axes[1].set_ylabel(r'$\mathrm{cond}_2(\rho_R)$ in left-canonical form')
for ax, panel in zip(axes, ['(a)', '(b)']):
    ax.set_xlabel('Optimization iteration')
    ax.set_xlim(0, 100)
    ax.grid(alpha=0.2)
    ax.set_title(panel, loc='left', fontsize=11)
fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=1, frameon=False)
fig.tight_layout(rect=(0, 0.21, 1, 1))
fig.savefig(out / 'first100-comparison.pdf', bbox_inches='tight')
fig.savefig(out / 'first100-comparison.png', dpi=180, bbox_inches='tight')
print(out)
