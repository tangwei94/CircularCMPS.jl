# multi-boson cMPS with tensor product parameterization
# The R matrices are parameterized as (I ⊗ ... ⊗ M ⊗ I ⊗ ... ⊗ I) 

struct MultiBosonCMPSData_tnp <: AbstractCMPSData 
    Q::MPSBondTensor
    Ms::Vector{<:MPSBondTensor}
end

function MultiBosonCMPS_tnp(f, χ::Integer, d::Integer)
    Q = TensorMap(f, ComplexF64, ℂ^(χ^d), ℂ^(χ^d))
    Ms = MPSBondTensor{ComplexSpace}[]
    for ix in 1:d 
        push!(Ms, TensorMap(f, ComplexF64, ℂ^χ, ℂ^χ))
    end
    return MultiBosonCMPS_tnp(Q, Ms)
end

# operations on the data. not on the cMPS
Base.:+(ψ::MultiBosonCMPS_tnp, ϕ::MultiBosonCMPS_tnp) = MultiBosonCMPS_tnp(ψ.Q + ϕ.Q, ψ.Ms .+ ϕ.Ms)
Base.:-(ψ::MultiBosonCMPS_tnp, ϕ::MultiBosonCMPS_tnp) = MultiBosonCMPS_tnp(ψ.Q - ϕ.Q, ψ.Ms .- ϕ.Ms)
Base.:*(ψ::MultiBosonCMPS_tnp, x::Number) = MultiBosonCMPS_tnp(ψ.Q * x, ψ.Ms .* x)
Base.:*(x::Number, ψ::MultiBosonCMPS_tnp) = MultiBosonCMPS_tnp(ψ.Q * x, ψ.Ms .* x)
LinearAlgebra.dot(ψ1::MultiBosonCMPS_tnp, ψ2::MultiBosonCMPS_tnp) = dot(ψ1.Q, ψ2.Q) + sum(dot.(ψ1.Ms, ψ2.Ms))
Base.eltype(ψ::MultiBosonCMPS_tnp) = eltype(ψ.Q)
LinearAlgebra.norm(ψ::MultiBosonCMPS_tnp) = sqrt(norm(dot(ψ, ψ)))
#Base.vec(ψ::MultiBosonCMPS_tnp) = [vec(ψ.Q); vec(ψ.Ms)]

function LinearAlgebra.mul!(w::MultiBosonCMPS_tnp, v::MultiBosonCMPS_tnp, α)
    mul!(w.Q, v.Q, α)
    for (Mw, Mv) in zip(w.Ms, v.Ms)
        mul!(Mw, Mv, α)
    end
    return w
end
function LinearAlgebra.rmul!(v::MultiBosonCMPS_tnp, α)
    rmul!(v.Q, α)
    for M in v.Ms
        rmul!(M, α)
    end
    return v
end
function LinearAlgebra.axpy!(α, ψ1::MultiBosonCMPS_tnp, ψ2::MultiBosonCMPS_tnp)
    axpy!(α, ψ1.Q, ψ2.Q)
    for (M1, M2) in zip(ψ1.Ms, ψ2.Ms)
        axpy!(α, M1, M2)
    end
    return ψ2
end
function LinearAlgebra.axpby!(α, ψ1::MultiBosonCMPS_tnp, β, ψ2::MultiBosonCMPS_tnp)
    axpby!(α, ψ1.Q, β, ψ2.Q)
    for (M1, M2) in zip(ψ1.Ms, ψ2.Ms)
        axpby!(α, M1, β, M2)
    end
    return ψ2
end
function Base.similar(ψ::MultiBosonCMPS_tnp) 
    Q = similar(ψ.Q)
    Ms = [similar(M) for M in ψ.Ms]
    return MultiBosonCMPS_tnp(Q, Ms)
end
function randomize!(ψ::MultiBosonCMPS_tnp)
    randomize!(ψ.Q)
    for ix in 1:length(ψ.Ms)
        randomize!(ψ.Ms[ix])
    end
end
function Base.zero(ψ::MultiBosonCMPS_tnp) 
    Q = zero(ψ.Q)
    Ms = [zero(M) for M in ψ.Ms]
    return MultiBosonCMPS_tnp(Q, Ms)
end

@inline get_χ(ψ::MultiBosonCMPS_tnp) = dim(_firstspace(ψ.Ms[1]))
@inline get_d(ψ::MultiBosonCMPS_tnp) = length(ψ.Ms)
TensorKit.space(ψ::MultiBosonCMPS_tnp) = _firstspace(ψ.Q)

function CMPSData(ψ::MultiBosonCMPS_tnp)
    χ, d = get_χ(ψ), get_d(ψ)
    δ = isomorphism(ℂ^(χ^d), (ℂ^χ)^d)
    Rs = map(1:d) do ix
        Rops = repeat(MPSBondTensor[id(ℂ^χ)], d)
        Rops[ix] = ψ.Ms[ix]
        return δ * foldr(⊗, Rops) * δ'
    end
    return CMPSData(ψ.Q, Rs)
end
function MultiBosonCMPS_tnp(ψ::CMPSData)
    χ1, d = get_χ(ψ), get_d(ψ)
    χ = Int(round(χ1^(1/d)))
    δ = isomorphism(ℂ^(χ^d), (ℂ^χ)^d)
    Ms = map(1:d) do ix
        indicesl = [(3:ix+1); [-1]; (ix+2:d+1)]
        indicesr = [(3:ix+1); [-2]; (ix+2:d+1)]
        tmpM = @ncon([δ', ψ.Rs[ix], δ], [[indicesl; 1], [1, 2], [2; indicesr]])
        return permute(tmpM, (1,), (2,))
    end
    return MultiBosonCMPS_tnp(ψ.Q, Ms)
end
function ChainRulesCore.rrule(::Type{CMPSData}, ψ::MultiBosonCMPS_tnp)
    function CMPSData_pushback(∂ψn)
        return NoTangent(), MultiBosonCMPS_tnp(∂ψn) 
    end
    return CMPSData(ψ), CMPSData_pushback
end
function expand(ψ::MultiBosonCMPS_tnp, χ::Integer; perturb::Float64=1e-3)
    χ0, d = get_χ(ψ), get_d(ψ)
    if χ <= χ0
        @warn "new χ not bigger than χ0"
        return ψ
    end
    
    mask = similar(ψ.Q, ℂ^(χ^d) ← ℂ^(χ^d))
    fill_data!(mask, randn)
    mask = perturb * mask
    Q = copy(mask)
    Q.data[1:(χ0^d), 1:(χ0^d)] += ψ.Q.data
    ΛQ, _ = eigen(ψ.Q)
    for ix in (χ0^d)+1:(χ^d)
        Q.data[ix, ix] -= ΛQ[χ0^d, χ0^d] # suppress
    end

    Ms = MPSBondTensor[]
    for M0 in ψ.Ms
        M = similar(M0, ℂ^χ ← ℂ^χ)
        fill_data!(M, randn)
        M = perturb * M
        M.data[1:χ0, 1:χ0] += M0.data
        push!(Ms, M)
    end

    return MultiBosonCMPS_tnp(Q, Ms) 
end

function tangent_map(ψm::MultiBosonCMPS_tnp, Xm::MultiBosonCMPS_tnp, EL::MPSBondTensor, ER::MPSBondTensor, Kinv::AbstractTensorMap{T, S, 2, 2}) where {T,S}
    χ, d = get_χ(ψm), get_d(ψm)
    ψ = CMPSData(ψm)
    X = CMPSData(Xm)
    Id = id(ℂ^(χ^d))

    ER /= tr(EL * ER)

    K1 = K_permute(K_otimes(Id, X.Q) + sum(K_otimes.(ψ.Rs, X.Rs)))
    @tensor ER1[-1; -2] := Kinv[-1 4; 3 -2] * K1[3 2; 1 4] * ER[1; 2]
    @tensor EL1[-1; -2] := Kinv[4 -1; -2 3] * K1[2 3; 4 1] * EL[1; 2]
    @tensor singular = EL[1; 2] * K1[2 3; 4 1] * ER[4; 3]

    mapped_XQ = EL * ER1 + EL1 * ER + singular * EL * ER
    mapped_XRs = map(zip(ψ.Rs, X.Rs)) do (R, XR)
        EL * XR * ER + EL1 * R * ER + EL * R * ER1 + singular * EL * R * ER
    end

    return MultiBosonCMPS_tnp(CMPSData(mapped_XQ, mapped_XRs)) 
end