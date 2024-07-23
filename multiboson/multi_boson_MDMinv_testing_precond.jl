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
a = parse(Int, ARGS[1])
ϕ = expand(res_wp[1], χ; perturb=0.1^a)
ϕ = left_canonical(ϕ)

b = 3#parse(Int, ARGS[1])
precond_prefactor = 0.1^b
precond_power = 1#parse(Float64, ARGS[2])

println("doing calculation for $(χ)")

res_wp = ground_state(Hm, ϕ; do_preconditioning=true, maxiter=1000, precond_power=precond_power, precond_prefactor=precond_prefactor);
@save "tmpdata/expand$(a)_precond$(b)_$(precond_power)-$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res_wp

println("with precond $(precond_prefactor) $(precond_power): E=$(res_wp[2]), gradnorm=$(norm(res_wp[3]))")
