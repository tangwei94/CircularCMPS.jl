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

Λs = sqrt(10) .^ (1:10)
χ = parse(Int, ARGS[1])
Δχ = parse(Int, ARGS[2])

χ = 4
Δχ = 0

################# computation ####################

ϕ1 = CMPSData(rand, χ, 2);
res_lm, _ = ground_state(Hm, ϕ1; Λs=Λs, gradtol=1e-6);
@save "multiboson/results/lagrangian_multiplier_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res_lm Λs
ϕ1 = res_lm[end][1]
ψ1 = MultiBosonCMPSData_MDMinv(ϕ1);

res = ground_state(Hm, ψ1; gradtol=1e-8, maxiter=1000); 
res_wop = ground_state(Hm, ψ1; do_preconditioning = false, gradtol=1e-8, maxiter=250); 
@save "multiboson/results/MDMinv_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res
@save "multiboson/results/MDMinv_wop_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res_wop

ψ2 = expand(res[1], 8, perturb = 1e-4);
ϕ2 = CMPSData(ψ2);
res_lm, _ = ground_state(Hm, ϕ2; Λs=sqrt(10) .^ (4:10), gradtol=1e-4);

ψ2 = MultiBosonCMPSData_MDMinv(res_lm[end][1]);
res = ground_state(Hm, ψ2; gradtol=1e-8, maxiter=1000); 
res_wop = ground_state(Hm, ψ2; do_preconditioning=false, gradtol=1e-8, maxiter=1000); 

ψ3 = expand(res[1], 16, perturb = 1e-4);
ϕ3 = CMPSData(ψ3);
res_lm, _ = ground_state(Hm, ϕ3; Λs=sqrt(10) .^ (4:10), gradtol=1e-4);

ψ3 = MultiBosonCMPSData_MDMinv(res_lm[end][1]);
res = ground_state(Hm, ψ3; gradtol=1e-8, maxiter=1000); 
res_wop = ground_state(Hm, ψ3; do_preconditioning=false, gradtol=1e-8, maxiter=1000); 

################# analysis ####################

@load "multiboson/results/lagrangian_multiplier_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ1).jld2" res_lm  
res1_lm = res_lm

################# outdated below ####################
@load "multiboson/results/preconditioned_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2).jld2" res1_lm res2_lm res3_lm
@load "multiboson/results/seperate_computation_$(c1)_$(c2)_$(μ1)_$(μ2).jld2" E_χ4 E_χ8
if abs(c12) < 1e-12
        Esep_χ4 = E_χ4
        Esep_χ8 = E_χ8 
else 
        Esep_χ4 = missing
        Esep_χ8 = missing
end
Es_lm, gnorms_lm = res3_lm[5][:, 1], res3_lm[5][:, 2]

fig = Figure(backgroundcolor = :white, fontsize=14, resolution= (400, 600))

gf = fig[1:5, 1] = GridLayout()
gl = fig[6, 1] = GridLayout()

ax1 = Axis(gf[1, 1], 
        xlabel = "steps",
        ylabel = "energy",
        )
lin1 = lines!(ax1, 1:length(Es_lm), Es_lm, label="w/ precond.")
lin2 = lines!(ax1, 1:length(Es_wop), Es_wop, label="w/o  precond.")
if !(Esep_χ4 isa Missing)
        lin3 = lines!(ax1, 1:length(Es_wop), fill(Esep_χ4, length(Es_wop)), linestyle=:dash, label="seperated χ=4")
        lin4 = lines!(ax1, 1:length(Es_wop), fill(Esep_χ8, length(Es_wop)), linestyle=:dot, label="seperated χ=8")
end
#axislegend(ax1, position=:rt)
@show fig

ax2 = Axis(gf[2, 1], 
        xlabel = "steps",
        ylabel = "gnorm",
        yscale = log10,
        )
lines!(ax2, 1:length(gnorms_lm), gnorms_lm, label="w/ precond.")
lines!(ax2, 1:length(gnorms_wop), gnorms_wop, label="w/o precond.")
#axislegend(ax2, position=:rb)
@show fig

Legend(gl[1, 1], ax1, nbanks=2)
@show fig
save("multiboson/results/result_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2).pdf", fig)