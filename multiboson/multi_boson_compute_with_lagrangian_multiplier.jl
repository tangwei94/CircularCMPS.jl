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

Hm = MultiBosonLiebLiniger([c1 c12; c12 c2], [μ1, μ2], Inf)

Λs = sqrt(10) .^ (4:10)
χ = parse(Int, ARGS[1])
Δχ = parse(Int, ARGS[2])

χ = 4
Δχ = 0

################# computation ####################

ϕ1 = left_canonical(CMPSData(rand, χ, 2))[2];
res_lm = ground_state(Hm, ϕ1; Λs=Λs, gradtol=1e-2, do_benchmark=true);
@save "multiboson/results/lagrangian_multiplier_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res_lm
ϕ1 = res_lm[1]
ψ1 = MultiBosonCMPSData_MDMinv(ϕ1);

res = ground_state(Hm, ψ1; gradtol=1e-8, maxiter=1000); 
res_wop = ground_state(Hm, ψ1; do_preconditioning = false, gradtol=1e-8, maxiter=1000); 
@save "multiboson/results/MDMinv_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res
@save "multiboson/results/MDMinv_wop_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res_wop

χ = 8
ψ2 = expand(res[1], χ, perturb = 1e-4);
ϕ2 = CMPSData(ψ2);
res_lm = ground_state(Hm, ϕ2; Λs=sqrt(10) .^ (4:10), gradtol=1e-2, do_benchmark=true);
@save "multiboson/results/lagrangian_multiplier_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res_lm
res_lm1 = ground_state(Hm, ϕ2; Λs=[1e5], gradtol=1e-2, maxiter=2000, do_benchmark=true);
@save "multiboson/results/lagrangian_multiplier_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ)-1.jld2" res_lm1
res_lm2 = ground_state(Hm, ϕ2; Λs=sqrt(10) .^ (4:10), gradtol=1e-6, maxiter=2000, do_benchmark=true);
@save "multiboson/results/lagrangian_multiplier_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ)-2.jld2" res_lm2

ψ2 = MultiBosonCMPSData_MDMinv(res_lm[1]);
res = ground_state(Hm, ψ2; gradtol=1e-8, maxiter=1000); 
res_wop = ground_state(Hm, ψ2; do_preconditioning=false, gradtol=1e-8, maxiter=1000); 
@save "multiboson/results/MDMinv_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res
@save "multiboson/results/MDMinv_wop_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res_wop

χ = 16
ψ3 = expand(res[1], χ, perturb = 1e-4);
ϕ3 = CMPSData(ψ3);
res_lm = ground_state(Hm, ϕ3; Λs=sqrt(10) .^ (4:10), gradtol=1e-3, do_benchmark=true);
@save "multiboson/results/lagrangian_multiplier_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res_lm

ψ3 = MultiBosonCMPSData_MDMinv(res_lm[1]);
res = ground_state(Hm, ψ3; gradtol=1e-8, maxiter=1000); 
res_wop = ground_state(Hm, ψ3; do_preconditioning=false, gradtol=1e-8, maxiter=1000); 
@save "multiboson/results/MDMinv_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res
@save "multiboson/results/MDMinv_wop_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res_wop

################# analysis ####################
χ = 16
@load "multiboson/results/MDMinv_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res
@load "multiboson/results/MDMinv_wop_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res_wop
Es, gnorms = res[5][:, 1], res[5][:, 2]
Es_wop, gnorms_wop = res_wop[5][:, 1], res_wop[5][:, 2]

fig = Figure(backgroundcolor = :white, fontsize=14, resolution= (400, 600))

gf = fig[1:5, 1] = GridLayout()
gl = fig[6, 1] = GridLayout()

ax1 = Axis(gf[1, 1], 
        xlabel = "steps",
        ylabel = "energy",
        )
lin1 = lines!(ax1, 1:length(Es), Es, label="w/ precond.")
lin2 = lines!(ax1, 1:length(Es_wop), Es_wop, label="w/o  precond.")
@show fig

ax2 = Axis(gf[2, 1], 
        xlabel = "steps",
        ylabel = "gnorm",
        yscale = log10,
        )
lines!(ax2, 1:length(gnorms), gnorms, label="w/ precond.")
lines!(ax2, 1:length(gnorms_wop), gnorms_wop, label="w/o precond.")
#axislegend(ax2, position=:rb)
@show fig

Legend(gl[1, 1], ax1, nbanks=2)
@show fig
save("multiboson/results/result_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).pdf", fig)