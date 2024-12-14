mutable struct MultiBosonCMPSData_tnp{T<:Number} <: AbstractCMPSData
    Q::Matrix{T}
    M::Matrix{T}
    Minv::Matrix{T}
    Bs::Vector{Matrix{T}}
    function MultiBosonCMPSData_tnp{T}(Q::Matrix{T}, M::Matrix{T}, Minv::Matrix{T}, Bs::Vector{Matrix{T}}) where T
        sizes_B = size.(Bs, 1)
        if !(size(Q, 1) == size(M, 1) == size(Minv, 1) == prod(sizes_B))
            throw(ArgumentError("size(Q, 1) == size(M, 1) == size(Minv, 1) == prod(sizes_B)"))
        end
        if norm(M * Minv - Matrix{T}(I, size(M))) > 1e-6
            @warn "M * Minv not close to I"
            Minv = inv(M)
        end
        return new{T}(Q, M, Minv, Bs)
    end
end

function MultiBosonCMPSData_tnp(Q::Matrix{T}, M::Matrix{T}, Minv::Matrix{T}, Bs::Vector{Matrix{T}}) where T
    return MultiBosonCMPSData_tnp{T}(Q, M, Minv, Bs)
end
function MultiBosonCMPSData_tnp(Q::Matrix{T}, M::Matrix{T}, Bs::Vector{Matrix{T}}) where T
    Minv = inv(M)
    return MultiBosonCMPSData_tnp{T}(Q, M, Minv, Bs)
end
function MultiBosonCMPSData_tnp(f, χb::Integer, d::Integer)
    χ = χb^d
    Q = f(ComplexF64, χ, χ)
    K = f(ComplexF64, χ, χ)
    M = exp(im * (K + K'))
    Minv = exp(-im * (K + K'))

    Bs = map(ix -> f(ComplexF64, χb, χb), 1:d)
    return MultiBosonCMPSData_tnp(Q, M, Minv, Bs) 
end

# operations on the data. not on the cMPS
Base.:+(ψ::MultiBosonCMPSData_tnp, ϕ::MultiBosonCMPSData_tnp) = MultiBosonCMPSData_tnp(ψ.Q + ϕ.Q, ψ.M + ϕ.M, ψ.Bs + ϕ.Bs)
Base.:-(ψ::MultiBosonCMPSData_tnp, ϕ::MultiBosonCMPSData_tnp) = MultiBosonCMPSData_tnp(ψ.Q - ϕ.Q, ψ.M - ϕ.M, ψ.Bs - ϕ.Bs)
Base.:*(ψ::MultiBosonCMPSData_tnp, x::Number) = MultiBosonCMPSData_tnp(ψ.Q * x, ψ.M * x, ψ.Minv / x, ψ.Bs * x)
Base.:*(x::Number, ψ::MultiBosonCMPSData_tnp) = MultiBosonCMPSData_tnp(ψ.Q * x, ψ.M * x, ψ.Minv / x, ψ.Bs * x)
Base.eltype(ψ::MultiBosonCMPSData_tnp) = eltype(ψ.Q)
LinearAlgebra.dot(ψ1::MultiBosonCMPSData_tnp, ψ2::MultiBosonCMPSData_tnp) = dot(ψ1.Q, ψ2.Q) + dot(ψ1.M, ψ2.M) + sum(dot.(ψ1.Bs, ψ2.Bs))
LinearAlgebra.norm(ψ::MultiBosonCMPSData_tnp) = sqrt(norm(dot(ψ, ψ)))

function Base.similar(ψ::MultiBosonCMPSData_tnp) 
    Q = similar(ψ.Q)
    Bs = similar.(ψ.Bs)
    return MultiBosonCMPSData_tnp(Q, copy(ψ.M), copy(ψ.Minv), Bs)
end
function randomize!(ψ::MultiBosonCMPSData_tnp)
    T = eltype(ψ)
    map!(x -> rand(T), ψ.Q, ψ.Q)
    map!(x -> rand(T), ψ.M, ψ.M)
    ψ.Minv .= inv(ψ.M)
    for ix in eachindex(ψ.Bs)
        map!(x -> rand(T), ψ.Bs[ix], ψ.Bs[ix])
    end
    return ψ
end

@inline get_χ(ψ::MultiBosonCMPSData_tnp) = size(ψ.Q, 1)
@inline get_χb(ψ::MultiBosonCMPSData_tnp) = size(ψ.Bs[1], 1)
@inline get_d(ψ::MultiBosonCMPSData_tnp) = length(ψ.Bs)

function construct_full_block_matrix(B::Matrix{T}, d::Integer, ix::Integer) where T
    χb = size(B, 1)
    blks = [Matrix{T}(I, χb, χb) for _ in 1:d]
    blks[ix] .= B
    return foldr(kron, blks)
end
function construct_full_block_matrix(Bs::Vector{Matrix{T}}) where T
    d = length(Bs)
    return map(ix -> construct_full_block_matrix(Bs[ix], d, ix), 1:d)
end
function extract_block_matrix(A::Matrix{<:Number}, d::Integer, ix::Integer)
    χ = size(A, 1)
    χb = Int(round(χ^(1/d)))
    Ablks = reshape(A, Tuple(repeat([χb], 2*d)))

    indices = repeat(1:d, 2)
    indices[d+1-ix] = -1 
    indices[2*d+1-ix] = -2
    B = @ncon((Ablks, ), (indices, ))
    return B 
end
function extract_block_matrix(As::Vector{Matrix{T}}) where T<:Number
    d = length(As)
    return map(ix -> extract_block_matrix(As[ix], d, ix), 1:d)
end
function CMPSData(ψ::MultiBosonCMPSData_tnp)
    χ, d = get_χ(ψ), get_d(ψ)
    Q = TensorMap(ψ.Q, ℂ^χ, ℂ^χ)
    Rs = map(1:d) do ix
        A = construct_full_block_matrix(ψ.Bs[ix], d, ix)
        TensorMap(ψ.M * A * ψ.Minv, ℂ^χ, ℂ^χ)
    end
    return CMPSData(Q, Rs)
end

function ChainRulesCore.rrule(::Type{CMPSData}, ψ::MultiBosonCMPSData_tnp) 
    d = get_d(ψ)
    M, Minv = ψ.M, ψ.Minv 
    Bs = ψ.Bs 
    function CMPSData_pushback(∂ψ)
        ∂Q = ∂ψ.Q.data 
        ∂Bs = map(ix -> extract_block_matrix(M' * ∂ψ.Rs[ix].data * Minv', d, ix), 1:d)
        Cs = construct_full_block_matrix(Bs)
        ∂M = sum(map(1:d) do ix
            ∂ψ.Rs[ix].data * Minv' * Cs[ix]' - Minv' * Cs[ix]' * M' * ∂ψ.Rs[ix].data * Minv'
        end)
        return NoTangent(), MultiBosonCMPSData_tnp(∂Q, ∂M, ∂Bs) 
    end
    return CMPSData(ψ), CMPSData_pushback
end

function left_canonical(ψ::MultiBosonCMPSData_tnp)
    ψc = CMPSData(ψ)

    X, ψcl = left_canonical(ψc)
    Q = ψcl.Q.data 
    M = X.data * ψ.M 
    Minv = ψ.Minv * inv(X.data)

    return MultiBosonCMPSData_tnp(Q, M, Minv, deepcopy(ψ.Bs))
end
function right_env(ψ::MultiBosonCMPSData_tnp)
    # transfer matrix
    ψc = CMPSData(ψ)
    fK = transfer_matrix(ψc, ψc)
    
    # solve the fixed-point equation
    init = similar(ψc.Q, space(ψc.Q, 1)←space(ψc.Q, 1))
    randomize!(init);
    _, vls, _ = eigsolve(fK, init, 1, :LR)
    vl = vls[1]
    
    U, S, _ = svd(vl.data)
    return U * Diagonal(S) * U'
end

function retract_left_canonical(ψ::MultiBosonCMPSData_tnp{T}, α::Float64, dBs::Vector{Matrix{T}}, X::Matrix{T}) where T
    # check left canonical form 
    ψc = CMPSData(ψ)
    ϵ = norm(ψc.Q + ψc.Q' + sum([R' * R for R in ψc.Rs]))
    (ϵ > 1e-9) && @warn "your cmps has deviated from the left canonical form, err=$ϵ"

    Bs = ψ.Bs .+ α .* dBs
    #X[diagind(X)] .- tr(X) / size(X, 1) # make X traceless
    #M = exp(α * X) * ψ.M
    #Minv = ψ.Minv * exp(-α * X)
    M = ψ.M * exp(α * X)
    Minv = exp(-α * X) * ψ.Minv

    Rs = Ref(ψ.M) .* construct_full_block_matrix(ψ.Bs) .* Ref(ψ.Minv)  
    R1s = Ref(M) .* construct_full_block_matrix(Bs) .* Ref(Minv)  
    ΔRs = R1s .- Rs

    Q = ψ.Q - sum([R' * ΔR + 0.5 * ΔR' * ΔR for (R, ΔR) in zip(Rs, ΔRs)])

    return MultiBosonCMPSData_tnp(Q, M, Minv, Bs)
end

#function expand(ψ::MultiBosonCMPSData_tnp, χ::Integer; perturb::Float64=1e-3)
#    χ0, d = get_χ(ψ), get_d(ψ)
#    if χ <= χ0
#        @warn "new χ not bigger than χ0"
#        return ψ
#    end
#
#    Qd, Qu = eigen(ψ.Q)
#    _, Qminind = findmin(real.(Qd))
#    Q = diagm(vcat(Qd, fill(Qd[Qminind] - log(10), χ - χ0)))
#
#    # the norm of M and Minv can be very ill-conditioned, e.g., one of them is very small and the other is very large
#    α = norm(ψ.M)/norm(ψ.Minv)
#    M0 = ψ.M / sqrt(α)
#
#    M = Matrix{eltype(ψ)}(I, χ, χ)
#    M += rand(eltype(ψ), χ, χ) * perturb
#    M[1:χ0, 1:χ0] = inv(Qu) * M0
#
#    Bs = map(1:d) do ix
#        Diagonal(vcat(diag(ψ.Bs[ix]), fill(perturb, 1:χ-χ0)))
#    end
#
#    return MultiBosonCMPSData_tnp(Q, M, Bs) 
#end

struct MultiBosonCMPSData_tnp_Grad{T<:Number} <: AbstractCMPSData
    dBs::Vector{Matrix{T}}
    X::Matrix{T}
    #function MultiBosonCMPSData_tnp_Grad{T}(dBs::Vector{Diagonal{T, Vector{T}}}, X::Matrix{T}) where T
    #    X[diagind(X)] .= 0 # force the diagonal part to be zero
    #    return new{T}(dBs, X)
    #end
end
function MultiBosonCMPSData_tnp_Grad(dBs::Vector{Diagonal{T, Vector{T}}}, X::Matrix{T}) where T
    return MultiBosonCMPSData_tnp_Grad{T}(dBs, X)
end
function MultiBosonCMPSData_tnp_Grad(v::Vector{T}, χb::Int, d::Int) where T
    χ = χb^d
    dBs = map(ix -> reshape(v[(χb^2)*(ix-1)+1:(χb^2)*ix], χb, χb), 1:d)
    X = reshape(v[d*(χb^2)+1:end], χ, χ)
    return MultiBosonCMPSData_tnp_Grad{T}(dBs, X)
end

Base.:+(a::MultiBosonCMPSData_tnp_Grad, b::MultiBosonCMPSData_tnp_Grad) = MultiBosonCMPSData_tnp_Grad(a.dBs .+ b.dBs, a.X + b.X)
Base.:-(a::MultiBosonCMPSData_tnp_Grad, b::MultiBosonCMPSData_tnp_Grad) = MultiBosonCMPSData_tnp_Grad(a.dBs .- b.dBs, a.X - b.X)
Base.:*(a::MultiBosonCMPSData_tnp_Grad, x::Number) = MultiBosonCMPSData_tnp_Grad(a.dBs * x, a.X * x)
Base.:*(x::Number, a::MultiBosonCMPSData_tnp_Grad) = MultiBosonCMPSData_tnp_Grad(a.dBs * x, a.X * x)
Base.eltype(a::MultiBosonCMPSData_tnp_Grad) = eltype(a.X)
LinearAlgebra.dot(a::MultiBosonCMPSData_tnp_Grad, b::MultiBosonCMPSData_tnp_Grad) = sum(dot.(a.dBs, b.dBs)) + dot(a.X, b.X)
TensorKit.inner(a::MultiBosonCMPSData_tnp_Grad, b::MultiBosonCMPSData_tnp_Grad) = real(dot(a, b))
LinearAlgebra.norm(a::MultiBosonCMPSData_tnp_Grad) = sqrt(norm(dot(a, a)))
Base.similar(a::MultiBosonCMPSData_tnp_Grad) = MultiBosonCMPSData_tnp_Grad(similar.(a.dBs), similar(a.X))
Base.vec(a::MultiBosonCMPSData_tnp_Grad) = vcat(vec.(a.dBs)..., vec(a.X))

function randomize!(a::MultiBosonCMPSData_tnp_Grad)
    T = eltype(a)
    for ix in eachindex(a.dBs)
        map!(x -> rand(T), a.dBs[ix], a.dBs[ix])
    end
    map!(x -> rand(T), a.X, a.X)
    return a
end

function diff_to_grad(ψ::MultiBosonCMPSData_tnp, ∂ψ::MultiBosonCMPSData_tnp)
    Rs = map(C->ψ.M * C * ψ.Minv, construct_full_block_matrix(ψ.Bs))

    gBs = ∂ψ.Bs .- extract_block_matrix(Ref(ψ.M') .* Rs .* Ref(∂ψ.Q) .* Ref(ψ.Minv'))
    gX = ψ.M' * sum([- R * ∂ψ.Q * R' + R' * R * ∂ψ.Q for R in Rs]) * ψ.Minv' + ψ.M' * ∂ψ.M 
    #gX[diagind(gX)] .-= tr(gX) / size(gX, 1) # make X traceless
    return MultiBosonCMPSData_tnp_Grad(gBs, gX)
end

function tangent_map(ψ::MultiBosonCMPSData_tnp, g::MultiBosonCMPSData_tnp_Grad; ρR = nothing)
    if isnothing(ρR)
        ρR = right_env(ψ)
    end

    EL = ψ.M' * ψ.M
    ER = ψ.Minv * ρR * ψ.Minv'
    Cs = construct_full_block_matrix(ψ.Bs) 
    dCs = construct_full_block_matrix(g.dBs)
    Ms = [EL * (g.X * C - C * g.X + dC) * ER for (dC, C) in zip(dCs, Cs)] 

    X_mapped = sum([M * C' - C' * M for (M, C) in zip(Ms, Cs)])
    dBs_mapped = extract_block_matrix(Ms)

    return MultiBosonCMPSData_tnp_Grad(dBs_mapped, X_mapped)
end
