using LinearAlgebra, TensorKit, KrylovKit
using ChainRules, Zygote 
using CairoMakie
using JLD2 
using OptimKit
using Revise
using CircularCMPS

c1, μ1 = 1., 2.
c2, μ2 = 1.5, 2.5
c12 = 0.5
#c2, μ2 = parse(Float64, ARGS[1]), parse(Float64, ARGS[2])
#c12 = parse(Float64, ARGS[3]) 

Hm = MultiBosonLiebLiniger([c1 c12; c12 c2], [μ1, μ2], Inf);

χ = 16
χprev = 12
Δχ = 4
@load "multiboson/results/preconditioned_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res_wp
Es_wp = res_wp[5][:, 1]
gnorms_wp = res_wp[5][:, 2]

as = [1]
bs = [3]
exs = [3, 4, 5]

fig = Figure(backgroundcolor = :white, fontsize=14, size= (600, 900))

gf = fig[1:5, 1] = GridLayout()
gl = fig[6, 1] = GridLayout()

ax1 = Axis(gf[1, 1], 
        xlabel = "steps",
        ylabel = "energy",
        )
ax2 = Axis(gf[2, 1], 
        xlabel = "steps",
        ylabel = "gnorm",
        yscale = log10,
        )
ax3 = Axis(gf[3, 1], 
        xlabel = "steps",
        ylabel = "δ",
        yscale = log10,
        )
let res_wp0 = res_wp
    Es_wp = res_wp0[5][:, 1]
    gnorms_wp = res_wp0[5][:, 2]
    deltas = gnorms_wp .* 1e-3
    lines!(ax1, 1:length(Es_wp), Es_wp, label="δ=0.001*gn", color = :black)
    lines!(ax2, 1:length(gnorms_wp), gnorms_wp, color=:black)
    lines!(ax3, 1:length(deltas), deltas, color=:black)
    
    for a in as, b in bs, ex in exs
        @load "tmpdata/expand$(ex)_precond$(b)_$(a)-$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res_wp
        Es_wp = res_wp[5][:, 1]
        gnorms_wp = res_wp[5][:, 2]
        deltas = 0.1^b * gnorms_wp .^ a
        lines!(ax1, 1:length(Es_wp), Es_wp, label="ex $(ex) 1e$(-b) gn^$(a)")
        lines!(ax2, 1:length(gnorms_wp), gnorms_wp)
        lines!(ax3, 1:length(deltas), deltas)
    end
end
@show fig

Legend(gl[1, 1], ax1, nbanks=3)
@show fig
save("tmpdata/$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).pdf", fig)