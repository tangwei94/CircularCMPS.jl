using LinearAlgebra, TensorKit, KrylovKit 
using ChainRules, Zygote  
using CairoMakie 
using JLD2 
using OptimKit 
using Revise 
using CircularCMPS 

c1, μ1 = 1, 2 
c2, μ2 = 1, 2 
c12 = 0.5 

Hm = MultiBosonLiebLiniger([c1 c12; c12 c2], [μ1, μ2], Inf) 

χb = 2
χ = 4

################# computation ####################

ψ0 = MultiBosonCMPSData_MDMinv(rand, χ, 2);
ψ0 = left_canonical(ψ0);
ψ0.Ds[1]

lgΛmin, lgΛmax, steps = 2, 6, 5
ΔlgΛ = (lgΛmax - lgΛmin) / (steps - 1)
Λs = 10 .^ (lgΛmin:ΔlgΛ:lgΛmax)

# initialization with lagrange multipiler
ψ = left_canonical(CMPSData(ψ0))[2];
res_lm = ground_state(Hm, ψ; Λs = Λs, gradtol=1e-2, maxiter=1000, do_prerun=true);
#@save "multiboson/results/lagrangian_multiplier_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ)-$(lgΛmin)_$(lgΛmax)_$(steps).jld2" res_lm

ϕ = left_canonical(MultiBosonCMPSData_MDMinv(res_lm[1]));
res1 = ground_state(Hm, left_canonical(ϕ); gradtol=1e-6, maxiter=1000, preconditioner_type=1);

@load "P.jld2" P Pinv
sp = eigvals(Hermitian(P))[χ+1:end]
eigvals(Hermitian(Pinv))[χ+1:end]
s = eigvals(Hermitian(sqrt(P) * Pinv * sqrt(P)))[χ+1:end]

ψ1 = expand(res1[1], 2*χ);
ψ1 = left_canonical(CMPSData(ψ1))[2];

res_lm = ground_state(Hm, ψ1; Λs = Λs, gradtol=1e-2, maxiter=1000, do_prerun=true);
ϕ1 = left_canonical(MultiBosonCMPSData_MDMinv(res_lm[1]));
res2 = ground_state(Hm, ϕ1; gradtol=1e-6, maxiter=1000, preconditioner_type=1);

@load "P.jld2" P Pinv
sp = eigvals(Hermitian(P))[2*χ+1:end]
eigvals(Hermitian(Pinv))[2*χ+1:end]
s = eigvals(Hermitian(Pinv * sqrt(P)))[2*χ+1:end]

ψ2 = expand(res2[1], 3*χ);
ψ2 = left_canonical(CMPSData(ψ2))[2];

res_lm2 = ground_state(Hm, ψ2; Λs = Λs, gradtol=1e-2, maxiter=1000, do_prerun=true);
ϕ2 = left_canonical(MultiBosonCMPSData_MDMinv(res_lm2[1]));
res3 = ground_state(Hm, ϕ2; gradtol=1e-6, maxiter=1000, preconditioner_type=1);

ψ3 = expand(res3[1], 4*χ);
ψ3 = left_canonical(CMPSData(ψ3))[2];

res_lm3 = ground_state(Hm, ψ3; Λs = Λs, gradtol=1e-2, maxiter=1000, do_prerun=true);
ϕ3 = left_canonical(MultiBosonCMPSData_MDMinv(res_lm3[1]));
res4 = ground_state(Hm, ϕ3; gradtol=1e-6, maxiter=1000, preconditioner_type=1);


res1[2], norm(res1[3])
ψ1 = expand(res1[1], 2*χ);
res2 = ground_state(Hm, left_canonical(ψ1); gradtol=1e-6, maxiter=1000, preconditioner_type=1);
res2[2], norm(res2[3])

H = Hm
function fE_inf(ψ::MultiBosonCMPSData_MDMinv)
    ψn = CMPSData(ψ)
    OH = kinetic(ψn) + H.cs[1,1]* point_interaction(ψn, 1) + H.cs[2,2]* point_interaction(ψn, 2) + H.cs[1,2] * point_interaction(ψn, 1, 2) + H.cs[2,1] * point_interaction(ψn, 2, 1) - H.μs[1] * particle_density(ψn, 1) - H.μs[2] * particle_density(ψn, 2)
    TM = TransferMatrix(ψn, ψn)
    envL = permute(left_env(TM), (), (1, 2))
    envR = permute(right_env(TM), (2, 1), ()) 
    return real(tr(envL * OH * envR) / tr(envL * envR))
end

g = fE_inf'(ψ0)
g1 = CircularCMPS.diff_to_grad(ψ0, g)
ρR = right_env(ψ0)
cond(res1[1].M)

CircularCMPS.tangent_map1(ψ0, g1)
eigvals(ρR)
eigvals(res1[1].M)

ψ0.Ds[1] .= ψ0.Ds[2]

function tangent_map_preconditioner(ψ::MultiBosonCMPSData_MDMinv{T}, _g::MultiBosonCMPSData_MDMinv_Grad{T}; κ::Int = -1) where T

    ρR = right_env(ψ)
    χ = size(ψ.M, 1)
    d = length(ψ.Ds)

    EL = ψ.M' * ψ.M
    ER = ψ.Minv * ρR * ψ.Minv'

    ΔDs = map(1:d) do jx
        ΔD = zeros(ComplexF64, χ, χ)
        for ix in axes(ΔD, 1), iy in axes(ΔD, 2)
            ΔD[ix, iy] = (ix == iy) ? 1 : (ψ.Ds[jx][iy, iy] - ψ.Ds[jx][ix, ix]) 
        end
        ΔD
    end 
    
    Ms = map(1:d) do jx
        M = _g.X + ΔDs[jx]
        if κ == 1
            M .= M .* ΔDs[jx]
            M = EL * M * ER
            M .= M .* conj.(ΔDs[jx])
        else
            M .= M .* conj.(ΔDs[jx])
            M = inv(EL) * M * inv(ER)
            M .= M .* ΔDs[jx]
        end
        M
    end

    X_mapped = sum(Ms) / d
    dDs_mapped = Diagonal.(Ms)

    return MultiBosonCMPSData_MDMinv_Grad(dDs_mapped, X_mapped)
end

g2 = tangent_map_preconditioner(ψ0, g1; κ = 1);
g3 = tangent_map_preconditioner(ψ0, g2; κ = -1);

@load "preconditioner_P.jld2" P
P
eigvals(Hermitian(P))[13:end]
