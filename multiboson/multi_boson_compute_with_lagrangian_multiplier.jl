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
res_lm1 = ground_state(Hm, ϕ2; Λs=[1e5], gradtol=1e-2, maxiter=1000, do_benchmark=true);
@save "multiboson/results/lagrangian_multiplier_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ)-1.jld2" res_lm1
res_lm2 = ground_state(Hm, ϕ2; Λs=sqrt(10) .^ (4:10), gradtol=1e-4, maxiter=1000, do_benchmark=true);
@save "multiboson/results/lagrangian_multiplier_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ)-2.jld2" res_lm2

ψ2 = left_canonical(MultiBosonCMPSData_MDMinv(res_lm[1]));
res = ground_state(Hm, ψ2; gradtol=1e-8, maxiter=1000); 
res_wop = ground_state(Hm, ψ2; do_preconditioning=false, gradtol=1e-8, maxiter=1000); 
res_lagrange = ground_state(Hm, left_canonical(CMPSData(ψ2))[2]; Λs=sqrt(10) .^ (11:20), gradtol=1e-6, maxiter=250, do_benchmark=true); 
@save "multiboson/results/MDMinv_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res
@save "multiboson/results/MDMinv_wop_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res_wop
@save "multiboson/results/lagrange_multiplier_further_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res_lagrange

χ = 16
@load "multiboson/results/MDMinv_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_8.jld2" res
ψ3 = expand(res[1], χ, perturb = 1e-4);
ϕ3 = left_canonical(CMPSData(ψ3))[2];
res_lm = ground_state(Hm, ϕ3; Λs=sqrt(10) .^ (4:10), gradtol=1e-2, do_benchmark=true);
@save "multiboson/results/lagrangian_multiplier_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res_lm

ψ3 = MultiBosonCMPSData_MDMinv(res_lm[1]);
res = ground_state(Hm, ψ3; gradtol=1e-8, maxiter=1000); 
res_wop = ground_state(Hm, ψ3; do_preconditioning=false, gradtol=1e-8, maxiter=1000); 
@save "multiboson/results/MDMinv_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res
@save "multiboson/results/MDMinv_wop_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res_wop

################# analysis 1: benchmark MDMinv ####################
χ = 16
@load "multiboson/results/MDMinv_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res
@load "multiboson/results/MDMinv_wop_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res_wop
#@load "multiboson/results/lagrange_multiplier_further_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res_lagrange
Es, gnorms = res[5][:, 1], res[5][:, 2]
Es_wop, gnorms_wop = res_wop[5][:, 1], res_wop[5][:, 2]
#Es_lagrange, gnorms_lagrange = res_lagrange[5][:, 1], res_lagrange[5][:, 2]

fig = Figure(backgroundcolor = :white, fontsize=14, resolution= (600, 600))

gf = fig[1:5, 1] = GridLayout()
gl = fig[6, 1] = GridLayout()

ax1 = Axis(gf[1, 1], 
        xlabel = "steps",
        ylabel = "energy",
        )
lines!(ax1, 1:length(Es), Es, label="w/ precond.")
lines!(ax1, 1:length(Es_wop), Es_wop, label="w/o  precond.")
#lines!(ax1, 1:length(Es_lagrange), Es_lagrange, label="further increasing Λ")
@show fig

ax2 = Axis(gf[2, 1], 
        xlabel = "steps",
        ylabel = "gnorm",
        yscale = log10,
        )
lines!(ax2, 1:length(gnorms), gnorms, label="w/ precond.")
lines!(ax2, 1:length(gnorms_wop), gnorms_wop, label="w/o precond.")
#lines!(ax2, 1:length(gnorms_lagrange), gnorms_lagrange, label="further increasing Λ")
#axislegend(ax2, position=:rb)
@show fig

Legend(gl[1, 1], ax1, nbanks=2)
@show fig
save("multiboson/results/result_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).pdf", fig)

####### analysis 2: benchmark the Lagrange multiplier step ############
χ = 8
@load "multiboson/results/lagrangian_multiplier_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res_lm
@load "multiboson/results/lagrangian_multiplier_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ)-1.jld2" res_lm1
@load "multiboson/results/lagrangian_multiplier_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ)-2.jld2" res_lm2
Es, gnorms = res_lm[5][:, 1], res_lm[5][:, 2]
Es1, gnorms1 = res_lm1[5][:, 1], res_lm1[5][:, 2]
Es2, gnorms2 = res_lm2[5][:, 1], res_lm2[5][:, 2]

fig = Figure(backgroundcolor = :white, fontsize=14, resolution= (400, 600))

gf = fig[1:5, 1] = GridLayout()
gl = fig[6, 1] = GridLayout()

ax1 = Axis(gf[1, 1], 
        xlabel = "steps",
        ylabel = "energy",
        )
ylims!(ax1, -2.13, -2.00)
lines!(ax1, 1:length(Es), Es, label="Λ=1e2->1e5, tol=1e-2")
lines!(ax1, 1:length(Es1), Es1, label="Λ=1e5, tol=1e-2")
lines!(ax1, 1:length(Es2), Es2, label="Λ=1e2->1e5, tol=1e-4")
@show fig

ax2 = Axis(gf[2, 1], 
        xlabel = "steps",
        ylabel = "gnorm",
        yscale = log10,
        )
lines!(ax2, 1:length(gnorms),  gnorms, label="Λ=1e2->1e5, tol=1e-2")
lines!(ax2, 1:length(gnorms1), gnorms1, label="Λ=1e5, tol=1e-2")
lines!(ax2, 1:length(gnorms2), gnorms2, label="Λ=1e2->1e5, tol=1e-4")
#axislegend(ax2, position=:rb)
@show fig

Legend(gl[1, 1], ax1, nbanks=2)
@show fig
save("multiboson/results/result_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).pdf", fig)
