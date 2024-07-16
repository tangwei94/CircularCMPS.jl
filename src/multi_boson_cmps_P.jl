mutable struct MultiBosonCMPSData_P <: AbstractCMPSData 
    Q::MPSBondTensor
    Ms::Vector{<:MPSBondTensor}
end

function MultiBosonCMPSData_P(f, χ::Integer, d::Integer)
    Q = TensorMap(f, ComplexF64, ℂ^(χ^d), ℂ^(χ^d))
    Ms = MPSBondTensor{ComplexSpace}[]
    for ix in 1:d 
        push!(Ms, TensorMap(f, ComplexF64, ℂ^χ, ℂ^χ))
    end
    return MultiBosonCMPSData_P(Q, Ms)
end

# operations on the data. not on the cMPS
Base.:+(ψ::MultiBosonCMPSData_P, ϕ::MultiBosonCMPSData_P) = MultiBosonCMPSData_P(ψ.Q + ϕ.Q, ψ.Ms .+ ϕ.Ms)
Base.:-(ψ::MultiBosonCMPSData_P, ϕ::MultiBosonCMPSData_P) = MultiBosonCMPSData_P(ψ.Q - ϕ.Q, ψ.Ms .- ϕ.Ms)
Base.:*(ψ::MultiBosonCMPSData_P, x::Number) = MultiBosonCMPSData_P(ψ.Q * x, ψ.Ms .* x)
Base.:*(x::Number, ψ::MultiBosonCMPSData_P) = MultiBosonCMPSData_P(ψ.Q * x, ψ.Ms .* x)
LinearAlgebra.dot(ψ1::MultiBosonCMPSData_P, ψ2::MultiBosonCMPSData_P) = dot(ψ1.Q, ψ2.Q) + sum(dot.(ψ1.Ms, ψ2.Ms))
Base.eltype(ψ::MultiBosonCMPSData_P) = eltype(ψ.Q)
LinearAlgebra.dot(ψ1::MultiBosonCMPSData_P, ψ2::MultiBosonCMPSData_P) = dot(ψ1.Q, ψ2.Q) + dot(ψ1.Ms, ψ2.Ms)
LinearAlgebra.norm(ψ::MultiBosonCMPSData_P) = sqrt(norm(dot(ψ, ψ)))
#Base.vec(ψ::MultiBosonCMPSData_P) = [vec(ψ.Q); vec(ψ.Ms)]

function LinearAlgebra.mul!(w::MultiBosonCMPSData_P, v::MultiBosonCMPSData_P, α)
    mul!(w.Q, v.Q, α)
    for (Mw, Mv) in zip(w.Ms, v.Ms)
        mul!(Mw, Mv, α)
    end
    return w
end
function LinearAlgebra.rmul!(v::MultiBosonCMPSData_P, α)
    rmul!(v.Q, α)
    for M in v.Ms
        rmul!(M, α)
    end
    return v
end

function LinearAlgebra.axpy!(α, ψ1::MultiBosonCMPSData_P, ψ2::MultiBosonCMPSData_P)
    axpy!(α, ψ1.Q, ψ2.Q)
    for (M1, M2) in zip(ψ1.Ms, ψ2.Ms)
        axpy!(α, M1, M2)
    end
    return ψ2
end
function LinearAlgebra.axpby!(α, ψ1::MultiBosonCMPSData_P, β, ψ2::MultiBosonCMPSData_P)
    axpby!(α, ψ1.Q, β, ψ2.Q)
    for (M1, M2) in zip(ψ1.Ms, ψ2.Ms)
        axpby!(α, M1, β, M2)
    end
    return ψ2
end

function Base.similar(ψ::MultiBosonCMPSData_P) 
    Q = similar(ψ.Q)
    Ms = [similar(M) for M in ψ.Ms]
    return MultiBosonCMPSData_P(Q, Ms)
end

function randomize!(ψ::MultiBosonCMPSData_P)
    randomize!(ψ.Q)
    for ix in 1:length(ψ.Ms)
        randomize!(ψ.Ms[ix])
    end
end

function Base.zero(ψ::MultiBosonCMPSData_P) 
    Q = zero(ψ.Q)
    Ms = [zero(M) for M in ψ.Ms]
    return MultiBosonCMPSData_P(Q, Ms)
end

@inline get_χ(ψ::MultiBosonCMPSData_P) = dim(_firstspace(ψ.Ms[1]))
@inline get_d(ψ::MultiBosonCMPSData_P) = length(ψ.Ms)
TensorKit.space(ψ::MultiBosonCMPSData_P) = _firstspace(ψ.Q)

function CMPSData(ψ::MultiBosonCMPSData_P)
    χ, d = get_χ(ψ), get_d(ψ)
    δ = isomorphism(ℂ^(χ^d), (ℂ^χ)^d)
    Rs = map(1:d) do ix
        Rops = repeat(MPSBondTensor[id(ℂ^χ)], d)
        Rops[ix] = ψ.Ms[ix]
        return δ * foldr(⊗, Rops) * δ'
    end
    return CMPSData(ψ.Q, Rs)
end

function MultiBosonCMPSData_P(ψ::CMPSData)
    χ1, d = get_χ(ψ), get_d(ψ)
    χ = Int(round(χ1^(1/d)))
    δ = isomorphism(ℂ^(χ^d), (ℂ^χ)^d)
    Ms = map(1:d) do ix
        indicesl = [(3:ix+1); [-1]; (ix+2:d+1)]
        indicesr = [(3:ix+1); [-2]; (ix+2:d+1)]
        tmpM = @ncon([δ', ψ.Rs[ix], δ], [[indicesl; 1], [1, 2], [2; indicesr]])
        return permute(tmpM, (1,), (2,))
    end
    return MultiBosonCMPSData_P(ψ.Q, Ms)
end

function ChainRulesCore.rrule(::Type{CMPSData}, ψ::MultiBosonCMPSData_P)
    function CMPSData_pushback(∂ψn)
        return NoTangent(), MultiBosonCMPSData_P(∂ψn) 
    end
    return CMPSData(ψ), CMPSData_pushback
end

function expand(ψ::MultiBosonCMPSData_P, χ::Integer; perturb::Float64=1e-3)
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

    return MultiBosonCMPSData_P(Q, Ms) 
end

function tangent_map(ψm::MultiBosonCMPSData_P, Xm::MultiBosonCMPSData_P, EL::MPSBondTensor, ER::MPSBondTensor, Kinv::AbstractTensorMap{S, 2, 2}) where {S}
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

    return MultiBosonCMPSData_P(CMPSData(mapped_XQ, mapped_XRs)) 
end