using LinearAlgebra, TensorKit, KrylovKit
using ChainRules, Zygote 
using CairoMakie
using JLD2 
using OptimKit
using Revise
using CircularCMPS

c1, μ1 = 1., 2.
c2, μ2 = 1.5, 2.5

H1 = SingleBosonLiebLiniger(c1, μ1, Inf)
H2 = SingleBosonLiebLiniger(c2, μ2, Inf)

ϕ_χ4 = CMPSData(rand, 4, 1)

res1_χ4 = ground_state(H1, ϕ_χ4)
eigvals(convert(Array, res1_χ4[1].Rs[1]))


res1_χ8 = ground_state(H1, expand(res1_χ4[1], 8, 100))

res2_χ4 = ground_state(H2, ϕ_χ4)
res2_χ8 = ground_state(H2, expand(res2_χ4[1], 8, 100))

E1_χ4 = res1_χ4[2] 
E2_χ4 = res2_χ4[2] 
E1_χ8 = res1_χ8[2] 
E2_χ8 = res2_χ8[2] 

E11_χ4 = E1_χ4 + E1_χ4
E11_χ8 = E1_χ8 + E1_χ8
E12_χ4 = E1_χ4 + E2_χ4
E12_χ8 = E1_χ8 + E2_χ8

@save "multiboson/results/seperate_computation_$(c1)_$(c2)_$(μ1)_$(μ2).jld2" res1_χ4 res2_χ4 res1_χ8 res2_χ8 E_χ4=E12_χ4 E_χ8=E12_χ8
#@save "multiboson/results/seperate_computation_$(c1)_$(c1)_$(μ1)_$(μ1).jld2" res1_χ4 res2_χ4 res1_χ8 res2_χ8 E_χ4=E11_χ4 E_χ8=E11_χ8
