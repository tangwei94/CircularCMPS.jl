mutable struct MultiBosonCMPSData{T<:Number} <: AbstractCMPSData
    Q::Matrix{T}
    Λs::Matrix{T}
end

function MultiBosonCMPSData(f, χ::Integer, d::Integer)
    Q = f(ComplexF64, χ, χ)
    Λs = f(ComplexF64, χ, d)
    return MultiBosonCMPSData{ComplexF64}(Q, Λs)
end

function MultiBosonCMPSData(v::Vector{T}, χ::Integer, d::Integer) where T<:Number
    Q = reshape(v[1:χ^2], (χ, χ))
    Λs = reshape(v[χ^2+1:end], (χ, d))
    return MultiBosonCMPSData{T}(Q, Λs)
end

# operations on the data. not on the cMPS
Base.:+(ψ::MultiBosonCMPSData, ϕ::MultiBosonCMPSData) = MultiBosonCMPSData(ψ.Q + ϕ.Q, ψ.Λs + ϕ.Λs)
Base.:-(ψ::MultiBosonCMPSData, ϕ::MultiBosonCMPSData) = MultiBosonCMPSData(ψ.Q - ϕ.Q, ψ.Λs - ϕ.Λs)
Base.:*(ψ::MultiBosonCMPSData, x::Number) = MultiBosonCMPSData(ψ.Q * x, ψ.Λs * x)
Base.:*(x::Number, ψ::MultiBosonCMPSData) = MultiBosonCMPSData(ψ.Q * x, ψ.Λs * x)
Base.eltype(ψ::MultiBosonCMPSData) = eltype(ψ.Q)
LinearAlgebra.dot(ψ1::MultiBosonCMPSData, ψ2::MultiBosonCMPSData) = dot(ψ1.Q, ψ2.Q) + dot(ψ1.Λs, ψ2.Λs)
LinearAlgebra.norm(ψ::MultiBosonCMPSData) = sqrt(norm(dot(ψ, ψ)))
Base.vec(ψ::MultiBosonCMPSData) = [vec(ψ.Q); vec(ψ.Λs)]

function LinearAlgebra.mul!(w::MultiBosonCMPSData, v::MultiBosonCMPSData, α)
    mul!(w.Q, v.Q, α)
    mul!(w.Λs, v.Λs, α)
    return w
end
function LinearAlgebra.rmul!(v::MultiBosonCMPSData, α)
    rmul!(v.Q, α)
    rmul!(v.Λs, α)
    return v
end

function LinearAlgebra.axpy!(α, ψ1::MultiBosonCMPSData, ψ2::MultiBosonCMPSData)
    axpy!(α, ψ1.Q, ψ2.Q)
    axpy!(α, ψ1.Λs, ψ2.Λs)
    return ψ2
end
function LinearAlgebra.axpby!(α, ψ1::MultiBosonCMPSData, β, ψ2::MultiBosonCMPSData)
    axpby!(α, ψ1.Q, β, ψ2.Q)
    axpby!(α, ψ1.Λs, β, ψ2.Λs)
    return ψ2
end

function Base.similar(ψ::MultiBosonCMPSData) 
    Q = similar(ψ.Q)
    Λs = similar(ψ.Λs)
    return MultiBosonCMPSData(Q, Λs)
end

function randomize!(ψ::MultiBosonCMPSData)
    T = eltype(ψ)
    map!(x -> randn(T), ψ.Q, ψ.Q)
    map!(x -> randn(T), ψ.Λs, ψ.Λs)
end

function Base.zero(ψ::MultiBosonCMPSData) 
    Q = zero(ψ.Q)
    Λs = zero(ψ.Λs)
    return MultiBosonCMPSData(Q, Λs)
end

@inline get_χ(ψ::MultiBosonCMPSData) = size(ψ.Q, 1)
@inline get_d(ψ::MultiBosonCMPSData) = size(ψ.Λs, 2)
TensorKit.space(ψ::MultiBosonCMPSData) = ℂ^(get_χ(ψ))

function CMPSData(ψ::MultiBosonCMPSData)
    χ, d = get_χ(ψ), get_d(ψ)
    
    Q = TensorMap(ψ.Q, ℂ^χ, ℂ^χ)
    Rs = map(1:d) do ix 
        TensorMap(diagm(ψ.Λs[:, ix]), ℂ^χ, ℂ^χ)
    end
    return CMPSData(Q, Rs)
end

function MultiBosonCMPSData(ψ::CMPSData)
    χ, d = get_χ(ψ), get_d(ψ)

    Q = ψ.Q.data
    Λs = zeros(eltype(ψ.Q), χ, d)
    for ix in 1:d
        Λs[:, ix] = diag(ψ.Rs[ix].data)
    end
    return MultiBosonCMPSData(Q, Λs)
end

function ChainRulesCore.rrule(::Type{CMPSData}, ψ::MultiBosonCMPSData)
    function CMPSData_pushback(∂ψ)
        return NoTangent(), MultiBosonCMPSData(∂ψ)
    end
    return CMPSData(ψ), CMPSData_pushback
end

function expand(ψ::MultiBosonCMPSData, χ::Integer; perturb::Float64=1e-1)
    χ0, d = get_χ(ψ), get_d(ψ)
    if χ <= χ0
        @warn "new χ not bigger than χ0"
        return ψ
    end
    Q = 0.1 * randn(eltype(ψ), χ, χ)
    Q[1:χ0, 1:χ0] = ψ.Q

    Λs = perturb * randn(eltype(ψ), χ, d)
    Λs[1:χ0, 1:d] = ψ.Λs

    return MultiBosonCMPSData(Q, Λs) 
end

function tangent_map(ψm::MultiBosonCMPSData, Xm::MultiBosonCMPSData, EL::MPSBondTensor, ER::MPSBondTensor, Kinv::AbstractTensorMap{S, 2, 2}) where {S}
    χ = get_χ(ψm)
    ψ = CMPSData(ψm)
    X = CMPSData(Xm)
    Id = id(ℂ^χ)

    ER /= tr(EL * ER)

    K1 = K_permute(K_otimes(Id, X.Q) + sum(K_otimes.(ψ.Rs, X.Rs)))
    @tensor ER1[-1; -2] := Kinv[-1 4; 3 -2] * K1[3 2; 1 4] * ER[1; 2]
    @tensor EL1[-1; -2] := Kinv[4 -1; -2 3] * K1[2 3; 4 1] * EL[1; 2]
    @tensor singular = EL[1; 2] * K1[2 3; 4 1] * ER[4; 3]

    mapped_XQ = EL * ER1 + EL1 * ER + singular * EL * ER
    mapped_XRs = map(zip(ψ.Rs, X.Rs)) do (R, XR)
        EL * XR * ER + EL1 * R * ER + EL * R * ER1 + singular * EL * R * ER
    end

    return MultiBosonCMPSData(CMPSData(mapped_XQ, mapped_XRs)) 
end