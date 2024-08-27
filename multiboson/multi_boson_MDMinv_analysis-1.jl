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

χ = 64
lgΛmin, lgΛmax, steps = 2, 10, 81

# check the lagrange multiplier computation
@load "multiboson/results/lagrangian_multiplier_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ)-$(lgΛmin)_$(lgΛmax)_$(steps).jld2" res_lm

Es_lm, errs_lm = res_lm[5][:, 1], res_lm[5][:, 3]

fig = Figure(backgroundcolor = :white, fontsize=14, resolution= (400, 600))

gf = fig[1:5, 1] = GridLayout()
#gl = fig[6, 1] = GridLayout()

ax1 = Axis(gf[1, 1], 
        xlabel = "steps",
        ylabel = "energy",
        )
lin1 = lines!(ax1, 1:length(Es_lm), Es_lm, label="χ=$(χ)")
axislegend(ax1, position=:rb)
@show fig

ax2 = Axis(gf[2, 1], 
        xlabel = "steps",
        ylabel = "|[R1, R2]|",
        yscale = log10,
        )
lines!(ax2, 1:length(errs_lm), errs_lm, label="χ=$(χ)")
axislegend(ax2, position=:rt)
@show fig

save("multiboson/results/lagrangian_multiplier_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ)-$(lgΛmin)_$(lgΛmax)_$(steps).pdf", fig)

# check the optimization afterwards
@load "multiboson/results/preconditioned_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ)-$(lgΛmin)_$(lgΛmax)_$(steps).jld2" res_wp
Es_wp, gnorms_wp = res_wp[5][:, 1], res_wp[5][:, 2]

fig = Figure(backgroundcolor = :white, fontsize=14, resolution= (400, 600))

gf = fig[1:5, 1] = GridLayout()
#gl = fig[6, 1] = GridLayout()

ax1 = Axis(gf[1, 1], 
        xlabel = "steps",
        ylabel = "energy",
        )
lin1 = lines!(ax1, 1:length(Es_wp), Es_wp, label="χ=$(χ)")
axislegend(ax1, position=:rt)
@show fig

ax2 = Axis(gf[2, 1], 
        xlabel = "steps",
        ylabel = "gnorm",
        yscale = log10,
        )
lines!(ax2, 1:length(gnorms_wp), gnorms_wp, label="χ=$(χ)")
axislegend(ax2, position=:rt)
@show fig

#Legend(gl[1, 1], ax1, nbanks=2)
@show fig
save("multiboson/results/preconditioned_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ)-$(lgΛmin)_$(lgΛmax)_$(steps).pdf", fig)