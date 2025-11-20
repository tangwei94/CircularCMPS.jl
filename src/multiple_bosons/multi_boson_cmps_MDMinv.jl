"""
    MultiBosonCMPSData_MDMinv{T<:Number} <: AbstractCMPSData

A type representing a multi-boson continuous matrix product states (cMPS) with R matrices parameterized as M*D*Minv.

# Fields
- `Q::Matrix{T}`: The Q matrix of the cMPS
- `M::Matrix{T}`: The M matrix used in the MDMinv parameterization
- `Minv::Matrix{T}`: The inverse of M matrix
- `Ds::Vector{Diagonal{T, Vector{T}}}`: Vector of diagonal matrices 
"""
mutable struct MultiBosonCMPSData_MDMinv{T<:Number} <: AbstractCMPSData
    Q::Matrix{T}
    M::Matrix{T}
    Minv::Matrix{T}
    Ds::Vector{Diagonal{T, Vector{T}}}
    function MultiBosonCMPSData_MDMinv{T}(Q::Matrix{T}, M::Matrix{T}, Minv::Matrix{T}, Ds::Vector{Diagonal{T, Vector{T}}}) where T
        if !(M * Minv ≈ Matrix{T}(I, size(M))) 
            @warn "M * Minv not close to I"
            Minv = inv(M)
        end
        return new{T}(Q, M, Minv, Ds)
    end
end

"""
    MultiBosonCMPSData_MDMinv(Q::Matrix{T}, M::Matrix{T}, Minv::Matrix{T}, Ds::Vector{Diagonal{T, Vector{T}}}) where T

Construct a MultiBosonCMPSData_MDMinv with given Q, M, Minv matrices and diagonal matrices Ds.
Verifies that M * Minv ≈ I, and recomputes Minv if necessary.
"""
function MultiBosonCMPSData_MDMinv(Q::Matrix{T}, M::Matrix{T}, Minv::Matrix{T}, Ds::Vector{Diagonal{T, Vector{T}}}) where T
    return MultiBosonCMPSData_MDMinv{T}(Q, M, Minv, Ds)
end

"""
    MultiBosonCMPSData_MDMinv(Q::Matrix{T}, M::Matrix{T}, Ds::Vector{Diagonal{T, Vector{T}}}) where T

Construct a MultiBosonCMPSData_MDMinv by computing Minv from M.
"""
function MultiBosonCMPSData_MDMinv(Q::Matrix{T}, M::Matrix{T}, Ds::Vector{Diagonal{T, Vector{T}}}) where T
    Minv = inv(M)
    return MultiBosonCMPSData_MDMinv{T}(Q, M, Minv, Ds)
end

"""
    MultiBosonCMPSData_MDMinv(f, χ::Integer, d::Integer)

Initialize a random MultiBosonCMPSData_MDMinv with bond dimension χ and d boson species.
The function f is used to generate random matrices and vectors.
"""
function MultiBosonCMPSData_MDMinv(f, χ::Integer, d::Integer)
    Q = f(ComplexF64, χ, χ)
    M = Matrix{ComplexF64}(I, χ, χ)
    Minv = Matrix{ComplexF64}(I, χ, χ)
    Ds = map(ix -> Diagonal(rand(ComplexF64, χ)), 1:d)
    return MultiBosonCMPSData_MDMinv{ComplexF64}(Q, M, Minv, Ds)
end

# operations on the data. not on the cMPS
Base.:+(ψ::MultiBosonCMPSData_MDMinv, ϕ::MultiBosonCMPSData_MDMinv) = MultiBosonCMPSData_MDMinv(ψ.Q + ϕ.Q, ψ.M + ϕ.M, ψ.Ds + ϕ.Ds)
Base.:-(ψ::MultiBosonCMPSData_MDMinv, ϕ::MultiBosonCMPSData_MDMinv) = MultiBosonCMPSData_MDMinv(ψ.Q - ϕ.Q, ψ.M - ϕ.M, ψ.Ds - ϕ.Ds)
Base.:*(ψ::MultiBosonCMPSData_MDMinv, x::Number) = MultiBosonCMPSData_MDMinv(ψ.Q * x, ψ.M * x, ψ.Minv / x, ψ.Ds * x)
Base.:*(x::Number, ψ::MultiBosonCMPSData_MDMinv) = MultiBosonCMPSData_MDMinv(ψ.Q * x, ψ.M * x, ψ.Minv / x, ψ.Ds * x)
Base.eltype(ψ::MultiBosonCMPSData_MDMinv) = eltype(ψ.Q)
LinearAlgebra.dot(ψ1::MultiBosonCMPSData_MDMinv, ψ2::MultiBosonCMPSData_MDMinv) = dot(ψ1.Q, ψ2.Q) + dot(ψ1.M, ψ2.M) + sum(dot.(ψ1.Ds, ψ2.Ds))
LinearAlgebra.norm(ψ::MultiBosonCMPSData_MDMinv) = sqrt(norm(dot(ψ, ψ)))

function Base.similar(ψ::MultiBosonCMPSData_MDMinv) 
    Q = similar(ψ.Q)
    Ds = similar.(ψ.Ds)
    return MultiBosonCMPSData_MDMinv(Q, copy(ψ.M), copy(ψ.Minv), Ds)
end
function randomize!(ψ::MultiBosonCMPSData_MDMinv)
    T = eltype(ψ)
    map!(x -> rand(T), ψ.Q, ψ.Q)
    map!(x -> rand(T), ψ.M, ψ.M)
    ψ.Minv .= inv(ψ.M)
    for ix in eachindex(ψ.Ds)
        d = view(ψ.Ds[ix], diagind(ψ.Ds[ix]))
        map!(x -> rand(T), d, d)
    end
    return ψ
end

@inline get_χ(ψ::MultiBosonCMPSData_MDMinv) = size(ψ.Q, 1)
@inline get_d(ψ::MultiBosonCMPSData_MDMinv) = length(ψ.Ds)
TensorKit.space(ψ::MultiBosonCMPSData_MDMinv) = ℂ^(get_χ(ψ))

"""
    CMPSData(ψ::MultiBosonCMPSData_MDMinv)

Convert MultiBosonCMPSData_MDMinv to standard CMPSData format.
"""
function CMPSData(ψ::MultiBosonCMPSData_MDMinv)
    χ, d = get_χ(ψ), get_d(ψ)
    
    Q = TensorMap(ψ.Q, ℂ^χ, ℂ^χ)
    Rs = map(ψ.Ds) do D 
        TensorMap(ψ.M * D * ψ.Minv, ℂ^χ, ℂ^χ)
    end
    return CMPSData(Q, Rs)
end

"""
    MultiBosonCMPSData_MDMinv(ψ::CMPSData)

Convert CMPSData to MultiBosonCMPSData_MDMinv format. This function is deprecated.
"""
function MultiBosonCMPSData_MDMinv(ψ::CMPSData)
    Q = convert(Array, ψ.Q)
    _, M = eigen(convert(Array, ψ.Rs[1]))
    Minv = inv(M)
  
    D0s = map(R->Minv * convert(Array, R) * M, ψ.Rs)
    Ds = map(D->Diagonal(D), D0s)
    err2 = sum(norm.(Ds .- D0s) .^ 2)
    @info "convert CMPSData to MultiBosonCMPSData_MDMinv, err = $(sqrt(err2))"
    return MultiBosonCMPSData_MDMinv(Q, M, Minv, Ds)
end

function MultiBosonCMPSData_MDMinv(ψ::MultiBosonCMPSData_diag)
    χ, d = get_χ(ψ), get_d(ψ)
    
    Q = ψ.Q
    M = Matrix{ComplexF64}(I, χ, χ)
    Minv = Matrix{ComplexF64}(I, χ, χ)
    Ds = map(ix -> Diagonal(ψ.Λs[:, ix]), 1:d)
    return MultiBosonCMPSData_MDMinv(Q, M, Minv, Ds)
end
function MultiBosonCMPSData_diag(ψ::MultiBosonCMPSData_MDMinv)
    χ, d = get_χ(ψ), get_d(ψ)
    
    Q = ψ.Minv * ψ.Q * ψ.M
    Λs = zeros(ComplexF64, χ, d)
    for ix in 1:d
        Λs[:, ix] = diag(ψ.Ds[ix])
    end
    
    return MultiBosonCMPSData_diag(Q, Λs)
end

function ChainRulesCore.rrule(::Type{CMPSData}, ψ::MultiBosonCMPSData_MDMinv)
    M, Minv = ψ.M, ψ.Minv
    Ds = ψ.Ds
    function CMPSData_pushback(∂ψ)
        ∂Q = convert(Array, ∂ψ.Q)
        ∂Ds = map(∂R -> Diagonal(M' * convert(Array, ∂R) * Minv'), ∂ψ.Rs)
        ∂M = sum([convert(Array, ∂R) * Minv' * D' for (∂R, D) in zip(∂ψ.Rs, Ds)]) - 
             sum([Minv' * D' * M' * convert(Array, ∂R) * Minv' for (∂R, D) in zip(∂ψ.Rs, Ds)])
        return NoTangent(), MultiBosonCMPSData_MDMinv(∂Q, ∂M, ∂Ds) 
    end
    return CMPSData(ψ), CMPSData_pushback
end

"""
    left_canonical(ψ::MultiBosonCMPSData_MDMinv)

Transform the state into left canonical form where Q + Q' + ∑ᵢRᵢ'Rᵢ = 0.
"""
function left_canonical(ψ::MultiBosonCMPSData_MDMinv)
    ψc = CMPSData(ψ)

    X, ψcl = left_canonical(ψc)
    Q = convert(Array, ψcl.Q) 
    M = convert(Array, X) * ψ.M 
    Minv = ψ.Minv * inv(convert(Array, X))

    return MultiBosonCMPSData_MDMinv(Q, M, Minv, deepcopy(ψ.Ds))
end

"""
    right_env(ψ::MultiBosonCMPSData_MDMinv)

Compute the right environment matrix (fixed point of the transfer matrix).
"""
function right_env(ψ::MultiBosonCMPSData_MDMinv; init = missing, verbosity = 0)
    # transfer matrix
    ψc = CMPSData(ψ)
    fK = TransferMatrix(ψc, ψc)
    
    # solve the fixed-point equation
    vl = right_env(fK; init = init, verbosity = verbosity)
    
    #U, S, _ = svd(convert(Array, vl))
    #return U * Diagonal(S) * U'
    return vl
end

"""
    retract_left_canonical(ψ::MultiBosonCMPSData_MDMinv{T}, α::Float64, dDs::Vector{Diagonal{T, Vector{T}}}, X::Matrix{T}) where T

Retract the state in the left canonical form along the tangent direction specified by dDs and X with step size α.
Maintains the left canonical form constraint.
"""
function retract_left_canonical(ψ::MultiBosonCMPSData_MDMinv{T}, α::Float64, dDs::Vector{Diagonal{T, Vector{T}}}, X::Matrix{T}) where T
    # check left canonical form 
    ψc = CMPSData(ψ)
    ϵ = norm(ψc.Q + ψc.Q' + sum([R' * R for R in ψc.Rs]))
    (ϵ > 1e-10) && @warn "your cmps has deviated from the left canonical form, err=$ϵ"
   
    Ds = ψ.Ds .+ α .* dDs
    
    M = ψ.M * exp(α * X)
    Minv = exp(-α * X) * ψ.Minv

    #Id = Matrix{T}(I, size(X, 1), size(X, 1))
    #P = inv(Id - α * X / 2) * (Id + α * X / 2)
    #Pinv = inv(P)
    #M = ψ.M * P
    #Minv = Pinv * ψ.Minv

    Rs = [ψ.M * D0 * ψ.Minv for D0 in ψ.Ds] 
    R1s = [M * D * Minv for D in Ds] 
    ΔRs = R1s .- Rs

    Q = ψ.Q - sum([R' * ΔR + 0.5 * ΔR' * ΔR for (R, ΔR) in zip(Rs, ΔRs)])

    return MultiBosonCMPSData_MDMinv(Q, M, Minv, Ds)
end

function expand(ψ::MultiBosonCMPSData_MDMinv; perturb = 1e-3)
    χ0, d = get_χ(ψ), get_d(ψ)
    χ = 2 * χ0

    I2 = Diagonal(ones(Float64, 2))
    Ds = map(1:d) do ix
        kron(ψ.Ds[ix], I2) 
    end
    Q = kron(ψ.Q, I2) 
    M = kron(ψ.M, I2)
    Minv = kron(ψ.Minv, I2)
    R0s = [M * D * Minv for D in Ds]

    X = rand(ComplexF64, χ, χ)
    X = (X + X') / norm(X + X')

    M = M * exp(perturb * X)
    Minv = exp(-perturb * X) * Minv
    Ds = map(Ds) do D
        dD = Diagonal(rand(ComplexF64, χ))
        dD = (dD + dD') / norm(dD + dD')
        D + perturb * dD
    end
    ΔRs = [M * D * Minv - R0 for (D, R0) in zip(Ds, R0s)]
    V = sum(-[R0' * ΔR + 0.5 * ΔR' * ΔR for (R0, ΔR) in zip(R0s, ΔRs)])
    Q = Q + V

    return MultiBosonCMPSData_MDMinv(Q, M, Minv, Ds)
end

function direct_sum(A::Matrix, B::Matrix)
    m1, n1 = size(A)
    m2, n2 = size(B)
    result = zeros(eltype(A), m1 + m2, n1 + n2)
    result[1:m1, 1:n1] = A
    result[m1+1:end, n1+1:end] = B
    return result
end
function direct_sum(A::Diagonal, B::Diagonal)
    diag_A = A.diag
    diag_B = B.diag
    
    combined_diag = vcat(diag_A, diag_B)
    
    return Diagonal(combined_diag)
end

function expand(ψ::MultiBosonCMPSData_MDMinv, ψ1::MultiBosonCMPSData_MDMinv; perturb = 1e-3)
    χ0, d = get_χ(ψ), get_d(ψ)
    χ1 = get_χ(ψ1)
    χ = χ0 + χ1

    Ds = map(1:d) do ix
        direct_sum(ψ.Ds[ix], ψ1.Ds[ix])
    end
    Q = direct_sum(ψ.Q, ψ1.Q)
    Q[diagind(Q)[χ0+1:end]] .-= perturb
    M = direct_sum(ψ.M, ψ1.M)
    Minv = direct_sum(ψ.Minv, ψ1.Minv)
    R0s = [M * D * Minv for D in Ds]

    X = rand(ComplexF64, χ, χ)
    X = (X + X') / norm(X + X')

    M = M * exp(perturb * X)
    Minv = exp(-perturb * X) * Minv
    Ds = map(Ds) do D
        dD = Diagonal(rand(ComplexF64, χ))
        dD = (dD + dD') / norm(dD + dD')
        D + perturb * dD
    end
    #ΔRs = [M * D * Minv - R0 for (D, R0) in zip(Ds, R0s)]
    V = rand(ComplexF64, χ, χ)#sum(-[R0' * ΔR + 0.5 * ΔR' * ΔR for (R0, ΔR) in zip(R0s, ΔRs)])
    V = (V + V') / norm(V + V')
    Q = Q + perturb * V

    return MultiBosonCMPSData_MDMinv(Q, M, Minv, Ds)
end

"""
    MultiBosonCMPSData_MDMinv_Grad{T<:Number} <: AbstractCMPSData

A type representing the gradient of a multi-boson continuous matrix product state (cMPS) with R matrices parameterized as M*D*Minv.

# Fields
- `dDs::Vector{Diagonal{T, Vector{T}}}`: Gradient components for the diagonal matrices D
- `X::Matrix{T}`: Gradient component for the M matrix

# Type Parameters
- `T`: The numeric type used for the gradient components (e.g., Float64, ComplexF64)

# Retraction
The gradient is used in retractions as follows:
- D -> D + a * dD, where a is a small step size
- M -> M * exp(a * X), where exp is the matrix exponential
"""
struct MultiBosonCMPSData_MDMinv_Grad{T<:Number} <: AbstractCMPSData
    dDs::Vector{Diagonal{T, Vector{T}}}
    X::Matrix{T}
    function MultiBosonCMPSData_MDMinv_Grad{T}(dDs::Vector{Diagonal{T, Vector{T}}}, X::Matrix{T}) where T
        X[diagind(X)] .= 0 # force the diagonal part to be zero
        return new{T}(dDs, X)
    end
end

"""
    MultiBosonCMPSData_MDMinv_Grad(dDs::Vector{Diagonal{T, Vector{T}}}, X::Matrix{T}) where T
    MultiBosonCMPSData_MDMinv_Grad(v::Vector{T}, χ::Int, d::Int) where T

Constructors for MultiBosonCMPSData_MDMinv_Grad. The second form constructs from a vector representation
where the first χ*d elements are for dDs and the remaining for X.
"""
function MultiBosonCMPSData_MDMinv_Grad(dDs::Vector{Diagonal{T, Vector{T}}}, X::Matrix{T}) where T
    return MultiBosonCMPSData_MDMinv_Grad{T}(dDs, X)
end
function MultiBosonCMPSData_MDMinv_Grad(v::Vector{T}, χ::Int, d::Int) where T
    dDs = map(ix -> Diagonal(v[χ*(ix-1)+1:χ*ix]), 1:d)
    X = reshape(v[χ*d+1:end], χ, χ)
    return MultiBosonCMPSData_MDMinv_Grad{T}(dDs, X)
end

"""
    Base.:+(a::MultiBosonCMPSData_MDMinv_Grad, b::MultiBosonCMPSData_MDMinv_Grad)
    Base.:-(a::MultiBosonCMPSData_MDMinv_Grad, b::MultiBosonCMPSData_MDMinv_Grad)
    Base.:*(a::MultiBosonCMPSData_MDMinv_Grad, x::Number)
    Base.:*(x::Number, a::MultiBosonCMPSData_MDMinv_Grad)

Basic arithmetic operations for gradient vectors.
"""
Base.:+(a::MultiBosonCMPSData_MDMinv_Grad, b::MultiBosonCMPSData_MDMinv_Grad) = MultiBosonCMPSData_MDMinv_Grad(a.dDs .+ b.dDs, a.X + b.X)
Base.:-(a::MultiBosonCMPSData_MDMinv_Grad, b::MultiBosonCMPSData_MDMinv_Grad) = MultiBosonCMPSData_MDMinv_Grad(a.dDs .- b.dDs, a.X - b.X)
Base.:*(a::MultiBosonCMPSData_MDMinv_Grad, x::Number) = MultiBosonCMPSData_MDMinv_Grad(a.dDs * x, a.X * x)
Base.:*(x::Number, a::MultiBosonCMPSData_MDMinv_Grad) = MultiBosonCMPSData_MDMinv_Grad(a.dDs * x, a.X * x)

"""
    Base.eltype(a::MultiBosonCMPSData_MDMinv_Grad)
    LinearAlgebra.dot(a::MultiBosonCMPSData_MDMinv_Grad, b::MultiBosonCMPSData_MDMinv_Grad)
    TensorKit.inner(a::MultiBosonCMPSData_MDMinv_Grad, b::MultiBosonCMPSData_MDMinv_Grad)
    LinearAlgebra.norm(a::MultiBosonCMPSData_MDMinv_Grad)

Linear algebra operations for gradient vectors.
"""
Base.eltype(a::MultiBosonCMPSData_MDMinv_Grad) = eltype(a.X)
LinearAlgebra.dot(a::MultiBosonCMPSData_MDMinv_Grad, b::MultiBosonCMPSData_MDMinv_Grad) = sum(dot.(a.dDs, b.dDs)) + dot(a.X, b.X)
TensorKit.inner(a::MultiBosonCMPSData_MDMinv_Grad, b::MultiBosonCMPSData_MDMinv_Grad) = real(dot(a, b))
LinearAlgebra.norm(a::MultiBosonCMPSData_MDMinv_Grad) = sqrt(norm(dot(a, a)))
function LinearAlgebra.lmul!(a::Number, g::MultiBosonCMPSData_MDMinv_Grad)
    map(dD -> lmul!(a, dD), g.dDs)
    g.X .*= a
    return g
end
function LinearAlgebra.rmul!(g::MultiBosonCMPSData_MDMinv_Grad, a::Number)
    map(dD -> rmul!(dD, a), g.dDs)
    g.X .*= a
    return g
end
function LinearAlgebra.axpy!(a, x::MultiBosonCMPSData_MDMinv_Grad, y::MultiBosonCMPSData_MDMinv_Grad)
    map((xd, yd) -> axpy!(a, xd, yd), x.dDs, y.dDs)  # Handle diagonal components
    axpy!(a, x.X, y.X)  # Handle matrix component
    return y
end

function VectorInterface.scalartype(a::MultiBosonCMPSData_MDMinv_Grad)
    return eltype(a.X)
end
function VectorInterface.zerovector(a::MultiBosonCMPSData_MDMinv_Grad)
    return MultiBosonCMPSData_MDMinv_Grad(zerovector(a.dDs), zerovector(a.X))
end
function VectorInterface.zerovector!(a::MultiBosonCMPSData_MDMinv_Grad)
    zerovector!(a.dDs)
    zerovector!(a.X)
    return a
end
function VectorInterface.zerovector!!(a::MultiBosonCMPSData_MDMinv_Grad)
    zerovector!!(a.dDs)
    zerovector!!(a.X)
    return a
end
function VectorInterface.scale(a::MultiBosonCMPSData_MDMinv_Grad, α::Number)
    dDs = scale(a.dDs, α)
    X = scale(a.X, α)
    return MultiBosonCMPSData_MDMinv_Grad(dDs, X)
end
function VectorInterface.scale!(a::MultiBosonCMPSData_MDMinv_Grad, α::Number)
    scale!(a.dDs, α)
    scale!(a.X, α)
    return a
end
function VectorInterface.scale!!(a::MultiBosonCMPSData_MDMinv_Grad, α::Number)
    scale!!(a.dDs, α)
    scale!!(a.X, α)
    return a
end
function VectorInterface.add(a::MultiBosonCMPSData_MDMinv_Grad, b::MultiBosonCMPSData_MDMinv_Grad, α::Number=1, β::Number=1)
    dDs = VectorInterface.add(a.dDs, b.dDs, α, β)
    X = VectorInterface.add(a.X, b.X, α, β)
    return MultiBosonCMPSData_MDMinv_Grad(dDs, X)
end
function VectorInterface.add!(a::MultiBosonCMPSData_MDMinv_Grad, b::MultiBosonCMPSData_MDMinv_Grad, α::Number=1, β::Number=1)
    VectorInterface.add!(a.dDs, b.dDs, α, β)
    VectorInterface.add!(a.X, b.X, α, β)
    return a
end
function VectorInterface.add!!(a::MultiBosonCMPSData_MDMinv_Grad, b::MultiBosonCMPSData_MDMinv_Grad, α::Number=1, β::Number=1)
    VectorInterface.add!!(a.dDs, b.dDs, α, β)
    VectorInterface.add!!(a.X, b.X, α, β)
    return a
end

"""
    Base.similar(a::MultiBosonCMPSData_MDMinv_Grad)
    Base.vec(a::MultiBosonCMPSData_MDMinv_Grad)

Utility functions to create similar gradient vectors and convert to vector form.
"""
Base.similar(a::MultiBosonCMPSData_MDMinv_Grad) = MultiBosonCMPSData_MDMinv_Grad(similar.(a.dDs), similar(a.X))
Base.vec(a::MultiBosonCMPSData_MDMinv_Grad) = vcat(diag.(a.dDs)..., vec(a.X))

"""
    randomize!(a::MultiBosonCMPSData_MDMinv_Grad)

Fill a gradient vector with random values.
"""
function randomize!(a::MultiBosonCMPSData_MDMinv_Grad)
    T = eltype(a)
    for ix in eachindex(a.dDs)
        v = view(a.dDs[ix], diagind(a.dDs[ix]))
        map!(x -> rand(T), v, v)
    end
    map!(x -> rand(T), a.X, a.X)
    return a
end

"""
    diff_to_grad(ψ::MultiBosonCMPSData_MDMinv, ∂ψ::MultiBosonCMPSData_MDMinv)

Convert the partial derivative of a state to a gradient vector in the tangent space.
"""
function diff_to_grad(ψ::MultiBosonCMPSData_MDMinv, ∂ψ::MultiBosonCMPSData_MDMinv)
    Rs = map(D->ψ.M * D * ψ.Minv, ψ.Ds)

    gDs = [Diagonal(-ψ.M' * R * ∂ψ.Q * ψ.Minv' + ∂D) for (R, ∂D) in zip(Rs, ∂ψ.Ds)]
    gX = ψ.M' * sum([- R * ∂ψ.Q * R' + R' * R * ∂ψ.Q for R in Rs]) * ψ.Minv' + ψ.M' * ∂ψ.M 
    #gX[diagind(gX)] .-= tr(gX) / size(gX, 1) # make X traceless

    return MultiBosonCMPSData_MDMinv_Grad(gDs, gX)
end

"""
    tangent_map(ψ::MultiBosonCMPSData_MDMinv, g::MultiBosonCMPSData_MDMinv_Grad; ρR = nothing)

Apply the tangent map to a gradient vector. If ρR is not provided, it will be computed using right_env.
"""
function tangent_map(ψ::MultiBosonCMPSData_MDMinv, g::MultiBosonCMPSData_MDMinv_Grad; ρRmat = missing)
    if ismissing(ρRmat)
        ρR = right_env(ψ)
        U, S, _ = svd(convert(Array, ρR))
        ρRmat = U * Diagonal(S) * U'
    end

    EL = ψ.M' * ψ.M
    ER = ψ.Minv * ρRmat * ψ.Minv'
    Ms = [EL * (g.X * D - D * g.X + dD) * ER for (dD, D) in zip(g.dDs, ψ.Ds)] 

    X_mapped = sum([M * D' - D' * M for (M, D) in zip(Ms, ψ.Ds)])
    dDs_mapped = Diagonal.(Ms)

    return MultiBosonCMPSData_MDMinv_Grad(dDs_mapped, X_mapped)
end

# another preconditioner. does not work well. can somehow stabilize the optimization in the dilute limit. But the convergence is very slow.
function tangent_map_h(ψ::MultiBosonCMPSData_MDMinv, g::MultiBosonCMPSData_MDMinv_Grad; ρRmat = missing, coupling::Float64 = 0.0)
    if ismissing(ρRmat)
        ρR = right_env(ψ)
        U, S, _ = svd(convert(Array, ρR))
        ρRmat = U * Diagonal(S) * U'
    end

    EL = ψ.M' * ψ.M
    ER = ψ.Minv * ρRmat * ψ.Minv'
    D1, D2, dD1, dD2 = ψ.Ds[1], ψ.Ds[2], g.dDs[1], g.dDs[2]
    M11 = EL * (g.X * D1 * D1 - D1 * D1 * g.X + 2*D1 * dD1) * ER
    M22 = EL * (g.X * D2 * D2 - D2 * D2 * g.X + 2*D2 * dD2) * ER
    M12 = EL * (g.X * D1 * D2 - D1 * D2 * g.X + 2*D1 * dD2) * ER * coupling
    M21 = EL * (g.X * D2 * D1 - D2 * D1 * g.X + 2*D2 * dD1) * ER * coupling

    X_mapped = M11 * (D1 * D1)' - (D1 * D1)' * M11 + 
               M22 * (D2 * D2)' - (D2 * D2)' * M22 +
               M12 * (D1 * D2)' - (D1 * D2)' * M12 +
               M21 * (D2 * D1)' - (D2 * D1)' * M21

    dD1_mapped = Diagonal(M11) * 2*D1' + Diagonal(M21) * 2*D2'
    dD2_mapped = Diagonal(M22) * 2*D2' + Diagonal(M12) * 2*D1'
    dDs_mapped = [dD1_mapped, dD2_mapped]

    return MultiBosonCMPSData_MDMinv_Grad(dDs_mapped, X_mapped)
end

function precondition_map(ψ::MultiBosonCMPSData_MDMinv, g::MultiBosonCMPSData_MDMinv_Grad; ρRmat = missing, δ = 1e-10, α = 1.0, P = missing, maxiter = 1, tol = 1e-12)
    if ismissing(ρRmat)
        ρR = right_env(ψ)
        U, S, _ = svd(convert(Array, ρR))
        ρRmat = U * Diagonal(S) * U'
    end
    χ, d = get_χ(ψ), get_d(ψ)

    function _f(v::Vector)
        gx = MultiBosonCMPSData_MDMinv_Grad(v, χ, d)
        gx1 = MultiBosonCMPSData_MDMinv_Grad(δ * gx.dDs, δ * α * gx.X)

        #v = deepcopy(vec(gx))
        #v1 = deepcopy(vec(gx1))
        #v[1:d*χ] .*= δ
        #v[d*χ+1:end] .*= δ * α
        #@show norm(v - v1)

        Mgx = tangent_map(ψ, gx; ρRmat = ρRmat) + gx1
        if ismissing(P)
            return vec(Mgx)
        else
            return P \ vec(Mgx)
        end
    end
    if ismissing(P)
        g1 = vec(g)
    else
        g1 = P \ vec(g)
    end

    v_mapped, info = linsolve(_f, g1, g1; ishermitian = true, isposdef = true, verbosity = 0, tol=tol, maxiter=maxiter)
    g_mapped = MultiBosonCMPSData_MDMinv_Grad(v_mapped, χ, d)
    return g_mapped, info
end

function energy(H::MultiBosonLiebLiniger, ψ::MultiBosonCMPSData_MDMinv; init_envL = missing, init_envR = missing)
    ψn = CMPSData(ψ)
    OH = kinetic(ψn) + H.cs[1,1]* point_interaction(ψn, 1) + H.cs[2,2]* point_interaction(ψn, 2) + H.cs[1,2] * point_interaction(ψn, 1, 2) + H.cs[2,1] * point_interaction(ψn, 2, 1) - H.μs[1] * particle_density(ψn, 1) - H.μs[2] * particle_density(ψn, 2)
    TM = TransferMatrix(ψn, ψn)
    envL = permute(left_env(TM; init = init_envL), (), (1, 2))
    envR = permute(right_env(TM; init = init_envR), (2, 1), ()) 
    return real(tr(envL * OH * envR) / tr(envL * envR))
end
function energy(H::MultiBosonLiebLinigerWithPairing, ψ::MultiBosonCMPSData_MDMinv; init_envL = missing, init_envR = missing)
    ψn = CMPSData(ψ)
    OH = kinetic(ψn) + H.cs[1,1]* point_interaction(ψn, 1) + H.cs[2,2]* point_interaction(ψn, 2) + H.cs[1,2] * point_interaction(ψn, 1, 2) + H.cs[2,1] * point_interaction(ψn, 2, 1) - H.μs[1] * particle_density(ψn, 1) - H.μs[2] * particle_density(ψn, 2) + H.us[1] * pairing(ψn, 1) + H.us[2] * pairing(ψn, 2)
    TM = TransferMatrix(ψn, ψn)
    envL = permute(left_env(TM; init = init_envL), (), (1, 2))
    envR = permute(right_env(TM; init = init_envR), (2, 1), ()) 
    return real(tr(envL * OH * envR) / tr(envL * envR))
end

function particle_density(ψ::MultiBosonCMPSData_MDMinv, component_index::Int; init_envL = missing, init_envR = missing)
    ψn = CMPSData(ψ)
    ON = particle_density(ψn, component_index)
    TM = TransferMatrix(ψn, ψn)
    envL = permute(left_env(TM; init = init_envL), (), (1, 2))
    envR = permute(right_env(TM; init = init_envR), (2, 1), ()) 
    return real(tr(envL * ON * envR) / tr(envL * envR))
end

function ground_state(H::AbstractHamiltonian, ψ0::MultiBosonCMPSData_MDMinv; preconditioner_type::Int=1, maxiter::Int=10000, gradtol=1e-6, m_LBFGS::Int=8, _finalize! = (x, f, g, numiter) -> (x, f, g, numiter))
    if H.L < Inf
        error("finite size not implemented yet.")
    end

    function fgE(x::OptimState{MultiBosonCMPSData_MDMinv{T}}) where T
        ψ = x.data
        init_envL = id(space(x.ρR, 1))
        init_envR = x.ρR
        fE_inf = ψ -> energy(H, ψ; init_envL = init_envL, init_envR = init_envR)
        E, ∂ψ = withgradient(fE_inf, ψ)
        g = diff_to_grad(ψ, ∂ψ[1])
        return E, g
    end
    
    function inner(x, a::MultiBosonCMPSData_MDMinv_Grad, b::MultiBosonCMPSData_MDMinv_Grad)
        # be careful the cases with or without a factor of 2. depends on how to define the complex gradient
        return real(dot(a, b)) 
    end

    function retract(x::OptimState{MultiBosonCMPSData_MDMinv{T}}, dψ::MultiBosonCMPSData_MDMinv_Grad, α::Real) where T
        ψ = x.data
        ψ1 = retract_left_canonical(ψ, α, dψ.dDs, dψ.X)
        return OptimState(ψ1, x.preconditioner, x.ρR, x.prev, x.df), dψ
    end
    function scale!(dψ::MultiBosonCMPSData_MDMinv_Grad, α::Number)
        for ix in eachindex(dψ.dDs)
            dψ.dDs[ix] .= dψ.dDs[ix] * α
        end
        dψ.X .= dψ.X .* α
        return dψ
    end
    function add!(y::MultiBosonCMPSData_MDMinv_Grad, x::MultiBosonCMPSData_MDMinv_Grad, α::Number=1, β::Number=1)
        for ix in eachindex(y.dDs)
            VectorInterface.add!(y.dDs[ix], x.dDs[ix], α, β)
        end
        VectorInterface.add!(y.X, x.X, α, β)
        return y
    end
    # only for comparison
    function _no_precondition(x::OptimState{MultiBosonCMPSData_MDMinv{T}}, dψ::MultiBosonCMPSData_MDMinv_Grad) where T
        return dψ
    end
    
    function _precondition1(x::OptimState{MultiBosonCMPSData_MDMinv{T}}, dψ::MultiBosonCMPSData_MDMinv_Grad) where T
        ψ = x.data
        χ, d = get_χ(ψ), get_d(ψ)
        ρR = right_env(ψ)
        U, S, _ = svd(convert(Array, ρR))
        ρRmat = U * Diagonal(S) * U'

        if ismissing(x.preconditioner)
            δ = isnan(x.df) ? 1e-3 : max(x.df, 1e-12)
            α = isnan(x.prev) ? 1.0 : x.prev

            P = zeros(ComplexF64, χ^2+d*χ, χ^2+d*χ)

            blas_num_threads = LinearAlgebra.BLAS.get_num_threads()
            LinearAlgebra.BLAS.set_num_threads(1)

            Threads.@threads for ix in 1:(χ^2+d*χ)
                v = zeros(ComplexF64, χ^2+d*χ)
                v[ix] = 1
                g = MultiBosonCMPSData_MDMinv_Grad(v, χ, d)
                g1 = tangent_map(ψ, g; ρRmat = ρRmat)
                P[:, ix] = vec(g1)
            end 
            LinearAlgebra.BLAS.set_num_threads(blas_num_threads)
            Peigs = eigvals(P)
            χ = size(ψ.M, 1)
            @show real(Peigs[χ+1]), real(Peigs[end])
                        
            P[diagind(P)[1:d*χ]] .+= δ # dD
            P[diagind(P)[d*χ+1:end]] .+= δ * α # X

            x.preconditioner = qr(P)
        end
        vp = x.preconditioner \ vec(dψ)
        PG = MultiBosonCMPSData_MDMinv_Grad(vp, χ, d)

        return PG
    end
    function _precondition2(x::OptimState{MultiBosonCMPSData_MDMinv{T}}, dψ::MultiBosonCMPSData_MDMinv_Grad) where T
        ψ = x.data
        χ, d = get_χ(ψ), get_d(ψ)
        ρR = right_env(ψ)
        U, S, _ = svd(convert(Array, ρR))
        ρRmat = U * Diagonal(S) * U'

        δ = isnan(x.df) ? 1e-3 : max(x.df, 1e-12)
        α = isnan(x.prev) ? 1.0 : x.prev
        PG, _ = precondition_map(ψ, dψ; δ = δ, α = α, P = x.preconditioner, ρRmat = ρRmat)

        return PG
    end
    function _precondition3(x::OptimState{MultiBosonCMPSData_MDMinv{T}}, dψ::MultiBosonCMPSData_MDMinv_Grad) where T
        if ismissing(x.preconditioner)
            return _precondition1(x, dψ)
        else 
            ψ = x.data
            χ, d = get_χ(ψ), get_d(ψ)
            ρR = right_env(ψ)
            U, S, _ = svd(convert(Array, ρR))
            ρRmat = U * Diagonal(S) * U'

            δ = isnan(x.df) ? 1e-3 : x.df
            α = isnan(x.prev) ? 1.0 : x.prev
            precondition_tol = isnan(x.prev) ? 0.01*norm(dψ) : norm(dψ) * min(sqrt(δ), 1e-3) # TODO. fix this.
            #println("preconditioner3: |dψ| = $(norm(dψ)), δ = $(δ), α = $(α)")
            PG, info = precondition_map(ψ, dψ; δ = δ, α = α, P = x.preconditioner, maxiter = 1, tol = precondition_tol, ρRmat = ρRmat)
            if norm(info.residual) > precondition_tol
                @show info
                @info "will recompute preconditioner..."
                x.preconditioner = missing
                PG = _precondition1(x, dψ)
            end
            return PG
        end
    end

    transport!(v, x, d, α, xnew) = v

    function finalize!(x::OptimState{MultiBosonCMPSData_MDMinv{T}}, f, g, numiter) where T
        if preconditioner_type == 1 || preconditioner_type == 2
            x.preconditioner = missing 
        end

        _finalize!(x, f, g, numiter)

        α = norm(f) ^ (1/3)
        x.prev = α # prev now plays the role of α. change name. 

        #U, S, _ = svd(convert(Array, x.ρR))
        #ρRmat = U * Diagonal(S) * U'
        #Ng = tangent_map(x.data, g; ρRmat = ρRmat)
        #δ = norm(dot(g, Ng))
        g1 = MultiBosonCMPSData_MDMinv_Grad( α^(-2.5) * g.dDs, α^(-3) * g.X)
        δ = norm(g1) ^ 2
        x.df = δ # FIXME. df now plays the role of δ. change name.
        
        x.ρR = right_env(x.data; init = x.ρR, verbosity = 1)

        full_env = convert(Array, x.ρR)
        println("finalize: δ = $(δ), α = $(α), cond number = $(cond(full_env))")
        return x, f, g, numiter
    end

    optalg_LBFGS = LBFGS(m_LBFGS; maxiter=maxiter, gradtol=gradtol, acceptfirst=false, verbosity=2)

    # preconditioner type 0 is not doing any preconditioning.
    # only preconditioner type 1, 2 works.  
    # preconditioner type1 construct the preconditioner matrix explicitly, and solve its inverse by factorization O(χ^6)
    # preconditioner type 2 solve the preconditioner inverse by lssolve.. But in practice it is slower than type1 due to the large condition number of P.
    if preconditioner_type == 1
        @show "using preconditioner 1"
        precondition★ = _precondition1
    elseif preconditioner_type == 2
        @show "using preconditioner 2"
        precondition★ = _precondition2
    elseif preconditioner_type == 3
        @show "using preconditioner 3"
        precondition★ = _precondition3
    elseif preconditioner_type == 0
        @show "no precondition"
        precondition★ = _no_precondition
    else
        error("preconditioner type not supported")
    end

    x0 = OptimState(left_canonical(ψ0)) 
    x0.ρR = right_env(x0.data; init = missing, verbosity = 2)
    x1, E1, grad1, numfg1, history1 = optimize(fgE, x0, optalg_LBFGS; retract = retract,
                                    precondition = precondition★,
                                    inner = inner, transport! =transport!,
                                    scale! = scale!, add! = add!, finalize! = finalize!
                                    );

    res = (x1.data, E1, grad1, numfg1, history1)
    return res
end

