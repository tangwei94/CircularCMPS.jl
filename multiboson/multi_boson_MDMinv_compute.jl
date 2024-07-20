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

χ = parse(Int, ARGS[1])
if length(ARGS) > 1
    Δχ = parse(Int, ARGS[2])
    @load "multiboson/results/preconditioned_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ-Δχ).jld2" res_wp
    ϕ = expand(res_wp[1], χ; perturb=1e-3)
else
    ϕ = MultiBosonCMPSData_MDMinv(rand, χ, 2)
end
ϕ = left_canonical(ϕ)

println("doing calculation for $(χ)")

res_wp = ground_state(Hm, ϕ; do_preconditioning=true, maxiter=1000);
@save "multiboson/results/preconditioned_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res_wp

res_wop = ground_state(Hm, ϕ; do_preconditioning=false, maxiter=1000);
@save "multiboson/results/unpreconditioned_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res_wop

println("with precond: E=$(res_wp[2]), gradnorm=$(norm(res_wp[3]))")
println("without precond: E=$(res_wop[2]), gradnorm=$(norm(res_wop[3]))")

Es_wp, gnorms_wp = res_wp[5][:, 1], res_wp[5][:, 2]
Es_wop, gnorms_wop = res_wop[5][:, 1], res_wop[5][:, 2]

fig = Figure(backgroundcolor = :white, fontsize=14, resolution= (400, 600))

gf = fig[1:5, 1] = GridLayout()
gl = fig[6, 1] = GridLayout()

ax1 = Axis(gf[1, 1], 
        xlabel = "steps",
        ylabel = "energy",
        )
lin1 = lines!(ax1, 1:length(Es_wp), Es_wp, label="w/ precond.")
lin2 = lines!(ax1, 1:length(Es_wop), Es_wop, label="w/o  precond.")
#axislegend(ax1, position=:rt)
@show fig

ax2 = Axis(gf[2, 1], 
        xlabel = "steps",
        ylabel = "gnorm",
        yscale = log10,
        )
lines!(ax2, 1:length(gnorms_wp), gnorms_wp, label="w/ precond.")
lines!(ax2, 1:length(gnorms_wop), gnorms_wop, label="w/o precond.")
#axislegend(ax2, position=:rt)
@show fig

Legend(gl[1, 1], ax1, nbanks=2)
@show fig
save("multiboson/results/result_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).pdf", fig)