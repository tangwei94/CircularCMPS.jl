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

χ1 = 4;
ϕ1 = MultiBosonCMPSData_MDMinv(rand, χ1, 2);
ϕ1 = left_canonical(ϕ1);

res1_MDMinv = ground_state(Hm, ϕ1; do_preconditioning=false, maxiter=500);
res1_MDMinv_precond = ground_state(Hm, ϕ1; do_preconditioning=true, maxiter=500);

ϕ2 = expand(res1_MDMinv_precond[1], 8; perturb=1e-3);
ϕ2 = left_canonical(ϕ2);
res2_MDMinv = ground_state(Hm, ϕ2; do_preconditioning=false, maxiter=500);
res2_MDMinv_precond = ground_state(Hm, ϕ2; do_preconditioning=true, maxiter=500);

gn1 = norm(res2_MDMinv[3])
gn2 = norm(res2_MDMinv_precond[3])

E1 = res2_MDMinv[2]
E2 = res2_MDMinv_precond[2]
E1 - E2

#χ1, χ2, χ3 = 4, 8, 16

#ϕ1 = MultiBosonCMPSData(rand, χ1, 2)
#res1_wp = ground_state(Hm, ϕ1; do_preconditioning=true, maxiter=2500);

#ϕ2 = expand(res1_wp[1], χ2; perturb=1e-1)
#res2_wp = ground_state(Hm, ϕ2; do_preconditioning=true, maxiter=2500);

#ϕ3 = expand(res2_wp[1], χ3; perturb=1e-1)
#res3_wp = ground_state(Hm, ϕ3; do_preconditioning=true, maxiter=2500);

#@save "multiboson/results/preconditioned_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2).jld2" res1_wp res2_wp res3_wp
#@load "multiboson/results/preconditioned_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2).jld2" res1_wp res2_wp res3_wp

#res1_wop = ground_state(Hm, ϕ1; do_preconditioning=false, maxiter=2500);
#res2_wop = ground_state(Hm, ϕ2; do_preconditioning=false, maxiter=2500);
#res3_wop = ground_state(Hm, ϕ3; do_preconditioning=false, maxiter=2500);

#@save "multiboson/results/unpreconditioned_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2).jld2" res1_wop res2_wop res3_wop

#@load "multiboson/results/unpreconditioned_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2).jld2" res1_wop res2_wop res3_wop
#@load "multiboson/results/preconditioned_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2).jld2" res1_wp res2_wp res3_wp
#@load "multiboson/results/seperate_computation_$(c1)_$(c2)_$(μ1)_$(μ2).jld2" E_χ4 E_χ8
#if c12 == 0.0
        #Esep_χ4 = E_χ4
        #Esep_χ8 = E_χ8 
#else 
        #Esep_χ4 = missing
        #Esep_χ8 = missing
#end
#Es_wp, gnorms_wp = res3_wp[5][:, 1], res3_wp[5][:, 2]
#Es_wop, gnorms_wop = res3_wop[5][:, 1], res3_wop[5][:, 2]

#fig = Figure(backgroundcolor = :white, fontsize=14, resolution= (400, 600))

#gf = fig[1:5, 1] = GridLayout()
#gl = fig[6, 1] = GridLayout()

#ax1 = Axis(gf[1, 1], 
        #xlabel = "steps",
        #ylabel = "energy",
        #)
#lin1 = lines!(ax1, 1:length(Es_wp), Es_wp, label="w/ precond.")
#lin2 = lines!(ax1, 1:length(Es_wop), Es_wop, label="w/o  precond.")
#if !(Esep_χ4 isa Missing)
        #lin3 = lines!(ax1, 1:length(Es_wop), fill(Esep_χ4, length(Es_wop)), linestyle=:dash, label="seperated χ=4")
        #lin4 = lines!(ax1, 1:length(Es_wop), fill(Esep_χ8, length(Es_wop)), linestyle=:dot, label="seperated χ=8")
#end
##axislegend(ax1, position=:rt)
#@show fig

#ax2 = Axis(gf[2, 1], 
        #xlabel = "steps",
        #ylabel = "gnorm",
        #yscale = log10,
        #)
#lines!(ax2, 1:length(gnorms_wp), gnorms_wp, label="w/ precond.")
#lines!(ax2, 1:length(gnorms_wop), gnorms_wop, label="w/o precond.")
##axislegend(ax2, position=:rb)
#@show fig

#Legend(gl[1, 1], ax1, nbanks=2)
#@show fig
#save("multiboson/results/result_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2).pdf", fig)