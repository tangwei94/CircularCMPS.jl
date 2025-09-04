using LinearAlgebra, TensorKit, KrylovKit
using TensorKitAD, ChainRules, Zygote 
using CairoMakie
using JLD2 
using OptimKit
using Revise
using CircularCMPS

c1, μ1 = 1., 2.
#c2, μ2 = parse(Float64, ARGS[1]), parse(Float64, ARGS[2])
#c12 = parse(Float64, ARGS[3]) 

c2, μ2 = 1., 2.
c12 = 0.0

@load "multiboson/results/seperate_computation_$(c1)_$(c2)_$(μ1)_$(μ2).jld2" res1_χ4 res2_χ4 res1_χ8 res2_χ8 E_χ4 E_χ8

ϕ1 = res1_χ8[1]
ϕ2 = res2_χ8[1]

Λ1, U1 = eigen(ϕ1.Rs[1])
Q1 = inv(U1) * ϕ1.Q * U1

Λ2, U2 = eigen(ϕ2.Rs[1])
Q2 = inv(U2) * ϕ2.Q * U2

Q = zeros(ComplexF64, 16, 16)
Λs = zeros(ComplexF64, 16, 2)

Q[1:8, 1:8] = Q1.data 
Q[9:16, 9:16] = Q2.data

Q[1:8, 9:16] = rand(ComplexF64, 8, 8)
Q[9:16, 1:8] = rand(ComplexF64, 8, 8)

Λs[1:8, 1] = diag(Λ1.data)
Λs[9:16, 1] = diag(Λ1.data)

ψ = MultiBosonCMPSData(Q, Λs)
Hm = MultiBosonLiebLiniger([c1 c12; c12 c2], [μ1, μ2], Inf)

res = ground_state(Hm, ψ; do_preconditioning=true, maxiter=5000);

Q[1:8, 9:16] = 0.1*rand(ComplexF64, 8, 8)
Q[9:16, 1:8] = 0.1*rand(ComplexF64, 8, 8)
ψ1 = MultiBosonCMPSData(Q, Λs)

res1 = ground_state(Hm, ψ1; do_preconditioning=true, maxiter=5000);


Q[1:8, 9:16] = 1e-2*rand(ComplexF64, 8, 8)
Q[9:16, 1:8] = 1e-2*rand(ComplexF64, 8, 8)
ψ2 = MultiBosonCMPSData(Q, Λs)

res2 = ground_state(Hm, ψ2; do_preconditioning=true, maxiter=5000);
# didn't converge after 1800 steps. give up