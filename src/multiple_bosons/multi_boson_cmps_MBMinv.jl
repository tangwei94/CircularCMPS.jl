# multi-boson cMPS with MBMinv parameterization
# The R matrices are parameterized as M * (I ⊗ ... ⊗ B ⊗ I ⊗ ... ⊗ I) * Minv

mutable struct MultiBosonCMPSData_MBMinv{T<:Number} <: AbstractCMPSData
    Q::Matrix{T}
    M::Matrix{T}
    Minv::Matrix{T}
    Bs::Vector{Matrix{T}}
    function MultiBosonCMPSData_MBMinv{T}(Q::Matrix{T}, M::Matrix{T}, Minv::Matrix{T}, Bs::Vector{Matrix{T}}) where T
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

function MultiBosonCMPSData_MBMinv(Q::Matrix{T}, M::Matrix{T}, Minv::Matrix{T}, Bs::Vector{Matrix{T}}) where T
    return MultiBosonCMPSData_MBMinv{T}(Q, M, Minv, Bs)
end
function MultiBosonCMPSData_MBMinv(Q::Matrix{T}, M::Matrix{T}, Bs::Vector{Matrix{T}}) where T
    Minv = inv(M)
    return MultiBosonCMPSData_MBMinv{T}(Q, M, Minv, Bs)
end
function MultiBosonCMPSData_MBMinv(f, χb::Integer, d::Integer)
    χ = χb^d
    Q = f(ComplexF64, χ, χ)
    K = f(ComplexF64, χ, χ)
    M = exp(im * (K + K'))
    Minv = exp(-im * (K + K'))

    Bs = map(ix -> f(ComplexF64, χb, χb), 1:d)
    return MultiBosonCMPSData_MBMinv(Q, M, Minv, Bs) 
end

# operations on the data. not on the cMPS
Base.:+(ψ::MultiBosonCMPSData_MBMinv, ϕ::MultiBosonCMPSData_tnp) = MultiBosonCMPSData_tnp(ψ.Q + ϕ.Q, ψ.M + ϕ.M, ψ.Bs + ϕ.Bs)
Base.:-(ψ::MultiBosonCMPSData_MBMinv, ϕ::MultiBosonCMPSData_tnp) = MultiBosonCMPSData_tnp(ψ.Q - ϕ.Q, ψ.M - ϕ.M, ψ.Bs - ϕ.Bs)
Base.:*(ψ::MultiBosonCMPSData_MBMinv, x::Number) = MultiBosonCMPSData_tnp(ψ.Q * x, ψ.M * x, ψ.Minv / x, ψ.Bs * x)
Base.:*(x::Number, ψ::MultiBosonCMPSData_MBMinv) = MultiBosonCMPSData_tnp(ψ.Q * x, ψ.M * x, ψ.Minv / x, ψ.Bs * x)
Base.eltype(ψ::MultiBosonCMPSData_MBMinv) = eltype(ψ.Q)
LinearAlgebra.dot(ψ1::MultiBosonCMPSData_MBMinv, ψ2::MultiBosonCMPSData_tnp) = dot(ψ1.Q, ψ2.Q) + dot(ψ1.M, ψ2.M) + sum(dot.(ψ1.Bs, ψ2.Bs))
LinearAlgebra.norm(ψ::MultiBosonCMPSData_MBMinv) = sqrt(norm(dot(ψ, ψ)))

function Base.similar(ψ::MultiBosonCMPSData_MBMinv) 
    Q = similar(ψ.Q)
    Bs = similar.(ψ.Bs)
    return MultiBosonCMPSData_MBMinv(Q, copy(ψ.M), copy(ψ.Minv), Bs)
end
function randomize!(ψ::MultiBosonCMPSData_MBMinv)
    T = eltype(ψ)
    map!(x -> rand(T), ψ.Q, ψ.Q)
    map!(x -> rand(T), ψ.M, ψ.M)
    ψ.Minv .= inv(ψ.M)
    for ix in eachindex(ψ.Bs)
        map!(x -> rand(T), ψ.Bs[ix], ψ.Bs[ix])
    end
    return ψ
end

@inline get_χ(ψ::MultiBosonCMPSData_MBMinv) = size(ψ.Q, 1)
@inline get_χb(ψ::MultiBosonCMPSData_MBMinv) = size(ψ.Bs[1], 1)
@inline get_d(ψ::MultiBosonCMPSData_MBMinv) = length(ψ.Bs)

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
function CMPSData(ψ::MultiBosonCMPSData_MBMinv)
    χ, d = get_χ(ψ), get_d(ψ)
    Q = TensorMap(ψ.Q, ℂ^χ, ℂ^χ)
    Rs = map(1:d) do ix
        A = construct_full_block_matrix(ψ.Bs[ix], d, ix)
        TensorMap(ψ.M * A * ψ.Minv, ℂ^χ, ℂ^χ)
    end
    return CMPSData(Q, Rs)
end

function ChainRulesCore.rrule(::Type{CMPSData}, ψ::MultiBosonCMPSData_MBMinv) 
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
        return NoTangent(), MultiBosonCMPSData_MBMinv(∂Q, ∂M, ∂Bs) 
    end
    return CMPSData(ψ), CMPSData_pushback
end

function left_canonical(ψ::MultiBosonCMPSData_MBMinv)
    ψc = CMPSData(ψ)

    X, ψcl = left_canonical(ψc)
    Q = ψcl.Q.data 
    M = X.data * ψ.M 
    Minv = ψ.Minv * inv(X.data)

    return MultiBosonCMPSData_MBMinv(Q, M, Minv, deepcopy(ψ.Bs))
end
function right_env(ψ::MultiBosonCMPSData_MBMinv)
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

function retract_left_canonical(ψ::MultiBosonCMPSData_MBMinv{T}, α::Float64, dBs::Vector{Matrix{T}}, X::Matrix{T}) where T
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

    return MultiBosonCMPSData_MBMinv(Q, M, Minv, Bs)
end

#function expand(ψ::MultiBosonCMPSData_MBMinv, χ::Integer; perturb::Float64=1e-3)
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
#    return MultiBosonCMPSData_MBMinv(Q, M, Bs) 
#end

struct MultiBosonCMPSData_MBMinv_Grad{T<:Number} <: AbstractCMPSData
    dBs::Vector{Matrix{T}}
    X::Matrix{T}
    #function MultiBosonCMPSData_MBMinv_Grad{T}(dBs::Vector{Diagonal{T, Vector{T}}}, X::Matrix{T}) where T
    #    X[diagind(X)] .= 0 # force the diagonal part to be zero
    #    return new{T}(dBs, X)
    #end
end
function MultiBosonCMPSData_MBMinv_Grad(dBs::Vector{Diagonal{T, Vector{T}}}, X::Matrix{T}) where T
    return MultiBosonCMPSData_MBMinv_Grad{T}(dBs, X)
end
function MultiBosonCMPSData_MBMinv_Grad(v::Vector{T}, χb::Int, d::Int) where T
    χ = χb^d
    dBs = map(ix -> reshape(v[(χb^2)*(ix-1)+1:(χb^2)*ix], χb, χb), 1:d)
    X = reshape(v[d*(χb^2)+1:end], χ, χ)
    return MultiBosonCMPSData_MBMinv_Grad{T}(dBs, X)
end

Base.:+(a::MultiBosonCMPSData_MBMinv_Grad, b::MultiBosonCMPSData_MBMinv_Grad) = MultiBosonCMPSData_MBMinv_Grad(a.dBs .+ b.dBs, a.X + b.X)
Base.:-(a::MultiBosonCMPSData_MBMinv_Grad, b::MultiBosonCMPSData_MBMinv_Grad) = MultiBosonCMPSData_MBMinv_Grad(a.dBs .- b.dBs, a.X - b.X)
Base.:*(a::MultiBosonCMPSData_MBMinv_Grad, x::Number) = MultiBosonCMPSData_MBMinv_Grad(a.dBs * x, a.X * x)
Base.:*(x::Number, a::MultiBosonCMPSData_MBMinv_Grad) = MultiBosonCMPSData_MBMinv_Grad(a.dBs * x, a.X * x)
Base.eltype(a::MultiBosonCMPSData_MBMinv_Grad) = eltype(a.X)
LinearAlgebra.dot(a::MultiBosonCMPSData_MBMinv_Grad, b::MultiBosonCMPSData_MBMinv_Grad) = sum(dot.(a.dBs, b.dBs)) + dot(a.X, b.X)
TensorKit.inner(a::MultiBosonCMPSData_MBMinv_Grad, b::MultiBosonCMPSData_MBMinv_Grad) = real(dot(a, b))
LinearAlgebra.norm(a::MultiBosonCMPSData_MBMinv_Grad) = sqrt(norm(dot(a, a)))
Base.similar(a::MultiBosonCMPSData_MBMinv_Grad) = MultiBosonCMPSData_MBMinv_Grad(similar.(a.dBs), similar(a.X))
Base.vec(a::MultiBosonCMPSData_MBMinv_Grad) = vcat(vec.(a.dBs)..., vec(a.X))

function randomize!(a::MultiBosonCMPSData_MBMinv_Grad)
    T = eltype(a)
    for ix in eachindex(a.dBs)
        map!(x -> rand(T), a.dBs[ix], a.dBs[ix])
    end
    map!(x -> rand(T), a.X, a.X)
    return a
end

function diff_to_grad(ψ::MultiBosonCMPSData_MBMinv, ∂ψ::MultiBosonCMPSData_tnp)
    Rs = map(C->ψ.M * C * ψ.Minv, construct_full_block_matrix(ψ.Bs))

    gBs = ∂ψ.Bs .- extract_block_matrix(Ref(ψ.M') .* Rs .* Ref(∂ψ.Q) .* Ref(ψ.Minv'))
    gX = ψ.M' * sum([- R * ∂ψ.Q * R' + R' * R * ∂ψ.Q for R in Rs]) * ψ.Minv' + ψ.M' * ∂ψ.M 
    #gX[diagind(gX)] .-= tr(gX) / size(gX, 1) # make X traceless
    return MultiBosonCMPSData_MBMinv_Grad(gBs, gX)
end

function tangent_map(ψ::MultiBosonCMPSData_MBMinv, g::MultiBosonCMPSData_MBMinv_Grad; ρR = nothing)
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

    return MultiBosonCMPSData_MBMinv_Grad(dBs_mapped, X_mapped)
end

function ground_state(H::MultiBosonLiebLiniger, ψ0::MultiBosonCMPSData_MBMinv; do_preconditioning::Bool=true, maxiter::Int=10000, gradtol=1e-6, fϵ=(x->10*x), m_LBFGS::Int=8)
    if H.L < Inf
        error("finite size not implemented yet.")
    end

    function fE_inf(ψ::MultiBosonCMPSData_MBMinv)
        ψn = CMPSData(ψ)
        OH = kinetic(ψn) + H.cs[1,1]* point_interaction(ψn, 1) + H.cs[2,2]* point_interaction(ψn, 2) + H.cs[1,2] * point_interaction(ψn, 1, 2) + H.cs[2,1] * point_interaction(ψn, 2, 1) - H.μs[1] * particle_density(ψn, 1) - H.μs[2] * particle_density(ψn, 2)
        TM = TransferMatrix(ψn, ψn)
        envL = permute(left_env(TM), (), (1, 2))
        envR = permute(right_env(TM), (2, 1), ()) 
        return real(tr(envL * OH * envR) / tr(envL * envR))
    end
    
    function fgE(x::OptimState{MultiBosonCMPSData_MBMinv{T}}) where T
        ψ = x.data
        E, ∂ψ = withgradient(fE_inf, ψ)
        g = diff_to_grad(ψ, ∂ψ[1])
        return E, g
    end
    
    function inner(x, a::MultiBosonCMPSData_MBMinv_Grad, b::MultiBosonCMPSData_MBMinv_Grad)
        # be careful the cases with or without a factor of 2. depends on how to define the complex gradient
        return real(dot(a, b)) 
    end

    function retract(x::OptimState{MultiBosonCMPSData_MBMinv{T}}, dψ::MultiBosonCMPSData_MBMinv_Grad, α::Real) where T
        ψ = x.data
        ψ1 = retract_left_canonical(ψ, α, dψ.dBs, dψ.X)
        return OptimState(ψ1, missing, x.prev, x.df), dψ
    end
    function scale!(dψ::MultiBosonCMPSData_MBMinv_Grad, α::Number)
        for ix in eachindex(dψ.dBs)
            dψ.dBs[ix] .= dψ.dBs[ix] * α
        end
        dψ.X .= dψ.X .* α
        return dψ
    end
    function add!(y::MultiBosonCMPSData_MBMinv_Grad, x::MultiBosonCMPSData_MBMinv_Grad, α::Number=1, β::Number=1)
        for ix in eachindex(y.dBs)
            VectorInterface.add!(y.dBs[ix], x.dBs[ix], α, β)
        end
        VectorInterface.add!(y.X, x.X, α, β)
        return y
    end
    # only for comparison
    function _no_precondition(x::OptimState{MultiBosonCMPSData_MBMinv{T}}, dψ::MultiBosonCMPSData_MBMinv_Grad) where T
        return dψ
    end

    # linsolve is slow
    #function _precondition_linsolve(ψ0::MultiBosonCMPSData_MBMinv, dψ::MultiBosonCMPSData_MBMinv_Grad)
    #    ϵ = max(1e-12, min(1, norm(dψ)^1.5))
    #    χ, d = get_χ(ψ0), get_d(ψ0)
    #    function f_map(v)
    #        g = MultiBosonCMPSData_MBMinv_Grad(v, χ, d)
    #        return vec(tangent_map(ψ0, g)) + ϵ*v
    #    end

    #    vp, _ = linsolve(f_map, vec(dψ), rand(ComplexF64, χ*d+χ^2); maxiter=1000, ishermitian = true, isposdef = true, tol=ϵ)
    #    return MultiBosonCMPSData_MBMinv_Grad(vp, χ, d)
    #end
    function _precondition(x::OptimState{MultiBosonCMPSData_MBMinv{T}}, dψ::MultiBosonCMPSData_MBMinv_Grad) where T
        ψ = x.data
        χb, χ, d = get_χb(ψ), get_χ(ψ), get_d(ψ)

        if ismissing(x.preconditioner)
            #ϵ = fϵ(norm(dψ)^2)#
            ϵ = isnan(x.df) ? fϵ(norm(dψ)^2) : fϵ(x.df)
            ϵ = max(1e-12, ϵ)

            P = zeros(ComplexF64, χ^2+d*(χb^2), χ^2+d*(χb^2))

            blas_num_threads = LinearAlgebra.BLAS.get_num_threads()
            LinearAlgebra.BLAS.set_num_threads(1)
            Threads.@threads for ix in 1:χ^2+d*(χb^2)
                v = zeros(ComplexF64, χ^2+d*(χb^2))
                v[ix] = 1
                g = MultiBosonCMPSData_MBMinv_Grad(v, χb, d)
                g1 = tangent_map(ψ, g)
                P[:, ix] = vec(g1)
            end 
            LinearAlgebra.BLAS.set_num_threads(blas_num_threads)
            P[diagind(P)] .+= ϵ
            x.preconditioner = qr(P)
        end
        vp = x.preconditioner \ vec(dψ)
        PG = MultiBosonCMPSData_MBMinv_Grad(vp, χb, d)
        return PG
    end

    transport!(v, x, d, α, xnew) = v

    function finalize!(x::OptimState{MultiBosonCMPSData_MBMinv{T}}, f, g, numiter) where T
        @show x.df / norm(g), norm(x.data.M), norm(x.data.Minv)
        x.preconditioner = missing
        x.df = abs(f - x.prev)
        x.prev = f
        return x, f, g, numiter
    end

    optalg_LBFGS = LBFGS(m_LBFGS; maxiter=maxiter, gradtol=gradtol, acceptfirst=false, verbosity=2)

    if do_preconditioning
        @show "doing precondition"
        precondition1 = _precondition
    else
        @show "no precondition"
        precondition1 = _no_precondition
    end

    x0 = OptimState(left_canonical(ψ0)) # FIXME. needs to do it twice??
    x1, E1, grad1, numfg1, history1 = optimize(fgE, x0, optalg_LBFGS; retract = retract,
                                    precondition = precondition1,
                                    inner = inner, transport! =transport!,
                                    scale! = scale!, add! = add!, finalize! = finalize!
                                    );

    res = (x1.data, E1, grad1, numfg1, history1)
    return res
end
