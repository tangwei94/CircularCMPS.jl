using LinearAlgebra, TensorKit, KrylovKit
using TensorKitAD, ChainRules, Zygote 
using CairoMakie
using JLD2 
using OptimKit
using Revise
using CircularCMPS

c1, μ1 = 1., 2.
c2, μ2 = 1., 2.
c12 = 0.5
#c2, μ2 = parse(Float64, ARGS[1]), parse(Float64, ARGS[2])
#c12 = parse(Float64, ARGS[3]) 

Hm = MultiBosonLiebLiniger([c1 c12; c12 c2], [μ1, μ2], Inf)

χ1, χ2, χ3 = 2, 3, 4

ϕ1 = MultiBosonCMPSData_P(rand, χ1, 2)
res1_wp = ground_state(Hm, ϕ1; do_preconditioning=true, maxiter=5000);

ϕ2 = expand(res1_wp[1], χ2; perturb=1e-1)
res2_wp = ground_state(Hm, ϕ2; do_preconditioning=true, maxiter=5000);

ϕ3 = expand(res2_wp[1], χ3; perturb=1e-1)
res3_wp = ground_state(Hm, ϕ3; do_preconditioning=true, maxiter=5000);

@save "multiboson/results/P_preconditioned_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2).jld2" res1_wp res2_wp res3_wp

ϕ1 = MultiBosonCMPSData_P(rand, χ1, 2)
res1_wp = ground_state(Hm, ϕ1; do_preconditioning=true, maxiter=20000);

ϕ2 = MultiBosonCMPSData_P(rand, χ2, 2)
res2_wp = ground_state(Hm, ϕ2; do_preconditioning=true, maxiter=20000);

ϕ3 = MultiBosonCMPSData_P(rand, χ3, 2)
res3_wp = ground_state(Hm, ϕ3; do_preconditioning=true, maxiter=20000);

@save "multiboson/results/P_preconditioned_randinit_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2).jld2" res1_wp res2_wp res3_wp

@load "multiboson/results/P_preconditioned_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2).jld2" res1_wp res2_wp res3_wp
res1_wp_P, res2_wp_P, res3_wp_P = res1_wp, res2_wp, res3_wp
@load "multiboson/results/P_preconditioned_randinit_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2).jld2" res1_wp res2_wp res3_wp
res1_wp_P_randinit, res2_wp_P_randinit, res3_wp_P_randinit = res1_wp, res2_wp, res3_wp
@load "multiboson/results/preconditioned_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2).jld2" res1_wp res2_wp res3_wp
@load "multiboson/results/seperate_computation_$(c1)_$(c2)_$(μ1)_$(μ2).jld2" E_χ4 E_χ8
if c12 == 0.0
        Esep_χ4 = E_χ4
        Esep_χ8 = E_χ8 
else 
        Esep_χ4 = missing
        Esep_χ8 = missing
end
Es_wp, gnorms_wp = res3_wp[5][:, 1], res3_wp[5][:, 2]
Es_wp_P, gnorms_wp_P = res3_wp_P[5][:, 1], res3_wp_P[5][:, 2]
Es_wp_P_randinit, gnorms_wp_P_randinit = res3_wp_P_randinit[5][:, 1], res3_wp_P_randinit[5][:, 2]

fig = Figure(backgroundcolor = :white, fontsize=14, resolution= (400, 600))

gf = fig[1:5, 1] = GridLayout()
gl = fig[6, 1] = GridLayout()

ax1 = Axis(gf[1, 1], 
        xlabel = "steps",
        ylabel = "energy",
        )
lin1 = lines!(ax1, 100:length(Es_wp), Es_wp[100:end], label="diagonal R")
lin2 = lines!(ax1, 400:length(Es_wp_P_randinit), Es_wp_P_randinit[400:end], label="tnp R, rand init")
lin3 = lines!(ax1, 100:length(Es_wp_P), Es_wp_P[100:end], label="tnp R")
if !(Esep_χ4 isa Missing)
        N = length(Es_wp)
        lin3 = lines!(ax1, 1:N, fill(Esep_χ4, N), linestyle=:dash, label="seperated χ=4")
        lin4 = lines!(ax1, 1:N, fill(Esep_χ8, N), linestyle=:dot, label="seperated χ=8")
end
#axislegend(ax1, position=:rt)
@show fig

ax2 = Axis(gf[2, 1], 
        xlabel = "steps",
        ylabel = "gnorm",
        yscale = log10,
        )
lin1 = lines!(ax2, 100:length(gnorms_wp), gnorms_wp[100:end], label="diagonal R")
lin2 = lines!(ax2, 200:length(gnorms_wp_P_randinit), gnorms_wp_P_randinit[200:end], label="tnp R, rand init")
lin3 = lines!(ax2, 100:length(gnorms_wp_P), gnorms_wp_P[100:end], label="tnp R")
#axislegend(ax2, position=:rb)
@show fig

Legend(gl[1, 1], ax1, nbanks=2)
@show fig
save("multiboson/results/P_result_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2).pdf", fig)