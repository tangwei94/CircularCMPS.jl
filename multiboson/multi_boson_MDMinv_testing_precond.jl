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
@load "multiboson/results/preconditioned_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ-Δχ).jld2" res_wp
ϕ = expand(res_wp[1], χ; perturb=1e-3)
ϕ = left_canonical(ϕ)

precond_power = parse(Float64, ARGS[1])

println("doing calculation for $(χ)")

res_wp = ground_state(Hm, ϕ; do_preconditioning=true, maxiter=1000, precond_power=precond_power);
@save "tmpdata/precond$(precond_power)_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res_wp

println("with precond power $(precond_power): E=$(res_wp[2]), gradnorm=$(norm(res_wp[3]))")

#Es_wp = res_wp[5][:, 1]
#
#fig = Figure(backgroundcolor = :white, fontsize=14, resolution= (400, 600))
#
#gf = fig[1:5, 1] = GridLayout()
#gl = fig[6, 1] = GridLayout()
#
#ax1 = Axis(gf[1, 1], 
#        xlabel = "steps",
#        ylabel = "energy",
#        )
#lin1 = lines!(ax1, 1:length(Es_wp), Es_wp, label="w/ precond. $(precond_power)")
##axislegend(ax1, position=:rt)
#@show fig
#
#ax2 = Axis(gf[2, 1], 
#        xlabel = "steps",
#        ylabel = "gnorm",
#        yscale = log10,
#        )
#lines!(ax2, 1:length(gnorms_wp), gnorms_wp, label="w/ precond. $(precond_power)")
##axislegend(ax2, position=:rt)
#@show fig
#
#Legend(gl[1, 1], ax1, nbanks=2)
#@show fig
#save("tmpdata/precond$(precond_power)_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).pdf", fig)