mutable struct MultiBosonCMPSData_MDMinv{T<:Number} <: AbstractCMPSData
    Q::Matrix{T}
    M::Matrix{T}
    Minv::Matrix{T}
    Λs::Matrix{T}
    function MultiBosonCMPSData_MDMinv(Q::Matrix{T}, M::Matrix{T}, Minv::Matrix{T}, Λs::Matrix{T}) where T
        if !(M * Minv ≈ Matrix{T}(I, size(M))) 
            @warn "M * Minv != I"
            Minv = inv(M)
        end
        return new(Q, M, Minv, Λs)
    end
    function MultiBosonCMPSData_MDMinv(Q::Matrix{T}, M::Matrix{T}, Λs::Matrix{T}) where T
        Minv = inv(M)
        return new(Q, M, Minv, Λs)
    end
    function MultiBosonCMPSData_MDMinv(f, χ::Integer, d::Integer)
        Q = f(ComplexF64, χ, χ)
        M = f(ComplexF64, χ, χ)
        Minv = inv(M)
        Λs = f(ComplexF64, χ, d)
        return MultiBosonCMPSData_MDMinv(Q, M, Minv, Λs)
    end
    function MultiBosonCMPSData_MDMinv(v::Vector{T}, χ::Integer, d::Integer) where T<:Number
        Q = reshape(v[1:χ^2], (χ, χ))
        M = reshape(v[χ^2+1:2*χ^2], (χ, χ))
        Minv = inv(M)
        Λs = reshape(v[2*χ^2+1:end], (χ, d))
        return MultiBosonCMPSData{T}(Q, M, Minv, Λs)
    end
end

# operations on the data. not on the cMPS
Base.:+(ψ::MultiBosonCMPSData_MDMinv, ϕ::MultiBosonCMPSData_MDMinv) = MultiBosonCMPSData_MDMinv(ψ.Q + ϕ.Q, ψ.M + ϕ.M, ψ.Λs + ϕ.Λs)
Base.:-(ψ::MultiBosonCMPSData_MDMinv, ϕ::MultiBosonCMPSData_MDMinv) = MultiBosonCMPSData_MDMinv(ψ.Q - ϕ.Q, ψ.M - ϕ.M, ψ.Λs - ϕ.Λs)
Base.:*(ψ::MultiBosonCMPSData_MDMinv, x::Number) = MultiBosonCMPSData_MDMinv(ψ.Q * x, ψ.M * x, ψ.Minv / x, ψ.Λs * x)
Base.:*(x::Number, ψ::MultiBosonCMPSData_MDMinv) = MultiBosonCMPSData_MDMinv(ψ.Q * x, ψ.M * x, ψ.Minv / x, ψ.Λs * x)
Base.eltype(ψ::MultiBosonCMPSData_MDMinv) = eltype(ψ.Q)
LinearAlgebra.dot(ψ1::MultiBosonCMPSData_MDMinv, ψ2::MultiBosonCMPSData_MDMinv) = dot(ψ1.Q, ψ2.Q) + dot(ψ1.M, ψ2.M) + dot(ψ1.Λs, ψ2.Λs)
LinearAlgebra.norm(ψ::MultiBosonCMPSData_MDMinv) = sqrt(norm(dot(ψ, ψ)))
Base.vec(ψ::MultiBosonCMPSData_MDMinv) = [vec(ψ.Q); vec(ψ.M); vec(ψ.Λs)]

# FIXME. I don't remember why I did this. Are these necessary?
function LinearAlgebra.mul!(w::MultiBosonCMPSData_MDMinv, v::MultiBosonCMPSData_MDMinv, α)
    mul!(w.Q, v.Q, α)
    mul!(w.M, v.M, α)
    w.Minv .= inv(w.M)
    mul!(w.Λs, v.Λs, α)
    return w
end
function LinearAlgebra.rmul!(v::MultiBosonCMPSData_MDMinv, α)
    rmul!(v.Q, α)
    rmul!(v.M, α)
    v.Minv .= inv(v.M)
    rmul!(v.Λs, α)
    return v
end
function LinearAlgebra.axpy!(α, ψ1::MultiBosonCMPSData_MDMinv, ψ2::MultiBosonCMPSData_MDMinv)
    axpy!(α, ψ1.Q, ψ2.Q)
    axpy!(α, ψ1.M, ψ2.M)
    ψ2.Minv .= inv(ψ2.M)
    axpy!(α, ψ1.Λs, ψ2.Λs)
    return ψ2
end
function LinearAlgebra.axpby!(α, ψ1::MultiBosonCMPSData_MDMinv, β, ψ2::MultiBosonCMPSData_MDMinv)
    axpby!(α, ψ1.Q, β, ψ2.Q)
    axpby!(α, ψ1.M, β, ψ2.M)
    ψ2.Minv .= inv(ψ2.M)
    axpby!(α, ψ1.Λs, β, ψ2.Λs)
    return ψ2
end

function Base.similar(ψ::MultiBosonCMPSData_MDMinv) 
    Q = similar(ψ.Q)
    Λs = similar(ψ.Λs)
    return MultiBosonCMPSData_MDMinv(Q, ψ.M, ψ.Minv, Λs)
end

function randomize!(ψ::MultiBosonCMPSData_MDMinv)
    T = eltype(ψ)
    map!(x -> randn(T), ψ.Q, ψ.Q)
    map!(x -> randn(T), ψ.M, ψ.M)
    ψ.Minv .= inv(ψ.M)
    map!(x -> randn(T), ψ.Λs, ψ.Λs)
end

function Base.zero(ψ::MultiBosonCMPSData_MDMinv) 
    Q = zero(ψ.Q)
    Λs = zero(ψ.Λs)
    return MultiBosonCMPSData_MDMinv(Q, ψ.M, ψ.Minv, Λs)
end

@inline get_χ(ψ::MultiBosonCMPSData_MDMinv) = size(ψ.Q, 1)
@inline get_d(ψ::MultiBosonCMPSData_MDMinv) = size(ψ.Λs, 2)
TensorKit.space(ψ::MultiBosonCMPSData_MDMinv) = ℂ^(get_χ(ψ))

function CMPSData(ψ::MultiBosonCMPSData_MDMinv)
    χ, d = get_χ(ψ), get_d(ψ)
    
    Q = TensorMap(ψ.Q, ℂ^χ, ℂ^χ)
    Rs = map(1:d) do ix 
        TensorMap(ψ.M * diagm(ψ.Λs[:, ix]) * ψ.Minv, ℂ^χ, ℂ^χ)
    end
    return CMPSData(Q, Rs)
end

function MultiBosonCMPSData_MDMinv(ψ::CMPSData)
    χ, d = get_χ(ψ), get_d(ψ)

    Q = ψ.Q.data
    Λs = zeros(eltype(ψ.Q), χ, d)
    _, M = eigen(ψ.Rs[1].data)
    Minv = inv(M)
   
    err2 = 0.
    for ix in 1:d
        D = Minv * ψ.Rs[ix].data * M
        Λs[:, ix] = diag(D)
        err2 += sum(norm.(D - diagm(diag(D))) .^ 2)
    end
    @info "convert CMPSData to MultiBosonCMPSData_MDMinv, err = $(sqrt(err2))"
    return MultiBosonCMPSData_MDMinv(Q, M, Minv, Λs)
end

function ChainRulesCore.rrule(::Type{CMPSData}, ψ::MultiBosonCMPSData_MDMinv)
    M, Minv = ψ.M, ψ.Minv
    d = get_d(ψ)
    Ds = map(1:d) do ix 
        TensorMap(diagm(ψ.Λs[:, ix]), ℂ^χ, ℂ^χ)
    end
    function CMPSData_pushback(∂ψ)
        ∂Q = ∂ψ.Q
        ∂Λs = map(∂ψ.Rs) do ∂R
            return M' * ∂R * Minv'
        end
        ∂M = sum(∂R .* adjoint.(Ds) * Minv') + sum(M' .* adjoint.(Ds) * Minv' * ∂R * Minv')
        return NoTangent(), (∂Q, ∂M, ∂Λs) 
    end
    return CMPSData(ψ), CMPSData_pushback
end

#function expand(ψ::MultiBosonCMPSData, χ::Integer; perturb::Float64=1e-1)
#    χ0, d = get_χ(ψ), get_d(ψ)
#    if χ <= χ0
#        @warn "new χ not bigger than χ0"
#        return ψ
#    end
#    Q = 0.1 * randn(eltype(ψ), χ, χ)
#    Q[1:χ0, 1:χ0] = ψ.Q
#
#    Λs = perturb * randn(eltype(ψ), χ, d)
#    Λs[1:χ0, 1:d] = ψ.Λs
#
#    return MultiBosonCMPSData(Q, Λs) 
#end

#function tangent_map(ψm::MultiBosonCMPSData, Xm::MultiBosonCMPSData, EL::MPSBondTensor, ER::MPSBondTensor, Kinv::AbstractTensorMap{S, 2, 2}) where {S<:EuclideanSpace}
#    χ = get_χ(ψm)
#    ψ = CMPSData(ψm)
#    X = CMPSData(Xm)
#    Id = id(ℂ^χ)
#
#    ER /= tr(EL * ER)
#
#    K1 = K_permute(K_otimes(Id, X.Q) + sum(K_otimes.(ψ.Rs, X.Rs)))
#    @tensor ER1[-1; -2] := Kinv[-1 4; 3 -2] * K1[3 2; 1 4] * ER[1; 2]
#    @tensor EL1[-1; -2] := Kinv[4 -1; -2 3] * K1[2 3; 4 1] * EL[1; 2]
#    @tensor singular = EL[1; 2] * K1[2 3; 4 1] * ER[4; 3]
#
#    mapped_XQ = EL * ER1 + EL1 * ER + singular * EL * ER
#    mapped_XRs = map(zip(ψ.Rs, X.Rs)) do (R, XR)
#        EL * XR * ER + EL1 * R * ER + EL * R * ER1 + singular * EL * R * ER
#    end
#
#    return MultiBosonCMPSData(CMPSData(mapped_XQ, mapped_XRs)) 
#end