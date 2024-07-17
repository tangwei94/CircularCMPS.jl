mutable struct MultiBosonCMPSData_MDMinv{T<:Number} <: AbstractCMPSData
    Q::Matrix{T}
    M::Matrix{T}
    Minv::Matrix{T}
    Ds::Vector{Diagonal{T, Vector{T}}}
    function MultiBosonCMPSData_MDMinv(Q::Matrix{T}, M::Matrix{T}, Minv::Matrix{T}, Ds::Vector{Diagonal{T, Vector{T}}}) where T
        if !(M * Minv ≈ Matrix{T}(I, size(M))) 
            @warn "M * Minv not close to I"
            Minv = pinv(M)
        end
        return new{T}(Q, M, Minv, Ds)
    end
    function MultiBosonCMPSData_MDMinv(Q::Matrix{T}, M::Matrix{T}, Ds::Vector{Diagonal{T, Vector{T}}}) where T
        Minv = pinv(M)
        return new{T}(Q, M, Minv, Ds)
    end
    function MultiBosonCMPSData_MDMinv(f, χ::Integer, d::Integer)
        Q = f(ComplexF64, χ, χ)
        M, _ = qr(f(ComplexF64, χ, χ))

        Minv = M'
        Ds = map(ix -> Diagonal(randn(ComplexF64, χ)), 1:d)
        return new{ComplexF64}(Q, M, Minv, Ds)
    end
end

# operations on the data. not on the cMPS
Base.:+(ψ::MultiBosonCMPSData_MDMinv, ϕ::MultiBosonCMPSData_MDMinv) = MultiBosonCMPSData_MDMinv(ψ.Q + ϕ.Q, ψ.M + ϕ.M, ψ.Ds + ϕ.Ds)
Base.:-(ψ::MultiBosonCMPSData_MDMinv, ϕ::MultiBosonCMPSData_MDMinv) = MultiBosonCMPSData_MDMinv(ψ.Q - ϕ.Q, ψ.M - ϕ.M, ψ.Ds - ϕ.Ds)
Base.:*(ψ::MultiBosonCMPSData_MDMinv, x::Number) = MultiBosonCMPSData_MDMinv(ψ.Q * x, ψ.M * x, ψ.Minv / x, ψ.Ds * x)
Base.:*(x::Number, ψ::MultiBosonCMPSData_MDMinv) = MultiBosonCMPSData_MDMinv(ψ.Q * x, ψ.M * x, ψ.Minv / x, ψ.Ds * x)
Base.eltype(ψ::MultiBosonCMPSData_MDMinv) = eltype(ψ.Q)
LinearAlgebra.dot(ψ1::MultiBosonCMPSData_MDMinv, ψ2::MultiBosonCMPSData_MDMinv) = dot(ψ1.Q, ψ2.Q) + dot(ψ1.M, ψ2.M) + sum(dot.(ψ1.Ds, ψ2.Ds))
LinearAlgebra.norm(ψ::MultiBosonCMPSData_MDMinv) = sqrt(norm(dot(ψ, ψ)))

#function LinearAlgebra.mul!(w::MultiBosonCMPSData_MDMinv, v::MultiBosonCMPSData_MDMinv, α)
#    mul!(w.Q, v.Q, α)
#    mul!(w.M, v.M, α)
#    w.Minv .= pinv(w.M)
#    mul!(w.Λs, v.Λs, α)
#    return w
#end
#function LinearAlgebra.rmul!(v::MultiBosonCMPSData_MDMinv, α)
#    rmul!(v.Q, α)
#    rmul!(v.M, α)
#    v.Minv .= pinv(v.M)
#    rmul!(v.Λs, α)
#    return v
#end
#function LinearAlgebra.axpy!(α, ψ1::MultiBosonCMPSData_MDMinv, ψ2::MultiBosonCMPSData_MDMinv)
#    axpy!(α, ψ1.Q, ψ2.Q)
#    axpy!(α, ψ1.M, ψ2.M)
#    ψ2.Minv .= pinv(ψ2.M)
#    axpy!(α, ψ1.Λs, ψ2.Λs)
#    return ψ2
#end
#function LinearAlgebra.axpby!(α, ψ1::MultiBosonCMPSData_MDMinv, β, ψ2::MultiBosonCMPSData_MDMinv)
#    axpby!(α, ψ1.Q, β, ψ2.Q)
#    axpby!(α, ψ1.M, β, ψ2.M)
#    ψ2.Minv .= pinv(ψ2.M)
#    axpby!(α, ψ1.Λs, β, ψ2.Λs)
#    return ψ2
#end

function Base.similar(ψ::MultiBosonCMPSData_MDMinv) 
    Q = similar(ψ.Q)
    Ds = similar.(ψ.Ds)
    return MultiBosonCMPSData_MDMinv(Q, copy(ψ.M), copy(ψ.Minv), Ds)
end
function randomize!(ψ::MultiBosonCMPSData_MDMinv)
    T = eltype(ψ)
    map!(x -> randn(T), ψ.Q, ψ.Q)
    map!(x -> rand(T), ψ.M, ψ.M)
    ψ.Minv .= pinv(ψ.M)
    for ix in eachindex(ψ.Ds)
        d = view(ψ.Ds[ix], diagind(ψ.Ds[ix]))
        map!(x -> randn(T), d, d)
    end
    return ψ
end

@inline get_χ(ψ::MultiBosonCMPSData_MDMinv) = size(ψ.Q, 1)
@inline get_d(ψ::MultiBosonCMPSData_MDMinv) = length(ψ.Ds)
TensorKit.space(ψ::MultiBosonCMPSData_MDMinv) = ℂ^(get_χ(ψ))

function CMPSData(ψ::MultiBosonCMPSData_MDMinv)
    χ, d = get_χ(ψ), get_d(ψ)
    
    Q = TensorMap(ψ.Q, ℂ^χ, ℂ^χ)
    Rs = map(ψ.Ds) do D 
        TensorMap(ψ.M * D * ψ.Minv, ℂ^χ, ℂ^χ)
    end
    return CMPSData(Q, Rs)
end

function MultiBosonCMPSData_MDMinv(ψ::CMPSData)
    Q = ψ.Q.data
    _, M = eigen(ψ.Rs[1].data)
    Minv = pinv(M)
  
    D0s = map(R->Minv * R.data * M, ψ.Rs)
    Ds = map(D->Diagonal(D), D0s)
    err2 = sum(norm.(Ds .- D0s) .^ 2)
    @info "convert CMPSData to MultiBosonCMPSData_MDMinv, err = $(sqrt(err2))"
    return MultiBosonCMPSData_MDMinv(Q, M, Minv, Ds)
end

function ChainRulesCore.rrule(::Type{CMPSData}, ψ::MultiBosonCMPSData_MDMinv)
    M, Minv = ψ.M, ψ.Minv
    Ds = ψ.Ds
    function CMPSData_pushback(∂ψ)
        ∂Q = ∂ψ.Q.data
        ∂Ds = map(∂R -> Diagonal(M' * ∂R.data * Minv'), ∂ψ.Rs)
        ∂M = sum([∂R.data * Minv' * D' for (∂R, D) in zip(∂ψ.Rs, Ds)]) - 
             sum([Minv' * D' * M' * ∂R.data * Minv' for (∂R, D) in zip(∂ψ.Rs, Ds)])
        return NoTangent(), MultiBosonCMPSData_MDMinv(∂Q, ∂M, ∂Ds) 
    end
    return CMPSData(ψ), CMPSData_pushback
end

function left_canonical(ψ::MultiBosonCMPSData_MDMinv)
    ψc = CMPSData(ψ)

    X, ψcl = left_canonical(ψc)
    Q = ψcl.Q.data 
    M = X.data * ψ.M 
    Minv = ψ.Minv * pinv(X.data)

    return MultiBosonCMPSData_MDMinv(Q, M, Minv, deepcopy(ψ.Ds))
end

function retract_left_canonical(ψ::MultiBosonCMPSData_MDMinv{T}, α::Float64, dDs::Vector{Diagonal{T, Vector{T}}}, X::Matrix{T}) where T
    Ds = ψ.Ds .+ α .* dDs
    M = exp(α * X) * ψ.M
    Minv = ψ.Minv * exp(-α * X)

    Rs = [ψ.M * D0 * ψ.Minv for D0 in ψ.Ds] 
    R1s = [M * D * Minv for D in Ds] 
    ΔRs = R1s .- Rs

    Q = ψ.Q - sum([R' * ΔR + 0.5 * ΔR' * ΔR for (R, ΔR) in zip(Rs, ΔRs)])

    return MultiBosonCMPSData_MDMinv(Q, M, Minv, Ds)
end

struct MultiBosonCMPSData_MDMinv_Grad{T<:Number} <: AbstractCMPSData
    dDs::Vector{Diagonal{T, Vector{T}}}
    X::Matrix{T}
end

Base.:+(a::MultiBosonCMPSData_MDMinv_Grad, b::MultiBosonCMPSData_MDMinv_Grad) = MultiBosonCMPSData_MDMinv_Grad(a.dDs .+ b.dDs, a.X + b.X)
Base.:-(a::MultiBosonCMPSData_MDMinv_Grad, b::MultiBosonCMPSData_MDMinv_Grad) = MultiBosonCMPSData_MDMinv_Grad(a.dDs .- b.dDs, a.X - b.X)
Base.:*(a::MultiBosonCMPSData_MDMinv_Grad, x::Number) = MultiBosonCMPSData_MDMinv_Grad(a.dDs * x, a.X * x)
Base.:*(x::Number, a::MultiBosonCMPSData_MDMinv_Grad) = MultiBosonCMPSData_MDMinv_Grad(a.dDs * x, a.X * x)
Base.eltype(a::MultiBosonCMPSData_MDMinv_Grad) = eltype(a.X)
LinearAlgebra.dot(a::MultiBosonCMPSData_MDMinv_Grad, b::MultiBosonCMPSData_MDMinv_Grad) = sum(dot.(a.dDs, b.dDs)) + dot(a.X, b.X)
LinearAlgebra.norm(a::MultiBosonCMPSData_MDMinv_Grad) = sqrt(norm(dot(a, a)))
Base.similar(a::MultiBosonCMPSData_MDMinv_Grad) = MultiBosonCMPSData_MDMinv_Grad(similar.(a.dDs), similar(a.X))
function randomize!(a::MultiBosonCMPSData_MDMinv_Grad)
    T = eltype(a)
    for ix in eachindex(a.dDs)
        v = view(a.dDs[ix], diagind(a.dDs[ix]))
        map!(x -> randn(T), v, v)
    end
    map!(x -> randn(T), a.X, a.X)
    return a
end

function diff_to_grad(ψ::MultiBosonCMPSData_MDMinv, ∂ψ::MultiBosonCMPSData_MDMinv)
    Rs = map(D->ψ.M * D * ψ.Minv, ψ.Ds)

    gDs = [Diagonal(-ψ.M' * R * ∂ψ.Q * ψ.Minv' + ∂D) for (R, ∂D) in zip(Rs, ∂ψ.Ds)]
    gX = sum([- R * ∂ψ.Q * R' + R' * R * ∂ψ.Q for R in Rs]) + ∂ψ.M * ψ.M'
    return MultiBosonCMPSData_MDMinv_Grad(gDs, gX)
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

#function tangent_map(ψm::MultiBosonCMPSData, Xm::MultiBosonCMPSData, EL::MPSBondTensor, ER::MPSBondTensor, Kinv::AbstractTensorMap{S, 2, 2}) where {S}
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