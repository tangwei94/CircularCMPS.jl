# multi-boson cMPS. The R matrices are parameterized as diagonal matrices

struct MultiBosonCMPSData_diag{T<:Number} <: AbstractCMPSData
    Q::Matrix{T}
    Λs::Matrix{T}
end

function MultiBosonCMPSData_diag(f, χ::Integer, d::Integer)
    Q = f(ComplexF64, χ, χ)
    Λs = f(ComplexF64, χ, d)
    return MultiBosonCMPSData_diag{ComplexF64}(Q, Λs)
end
function MultiBosonCMPSData_diag(v::Vector{T}, χ::Integer, d::Integer) where T<:Number
    Q = reshape(v[1:χ^2], (χ, χ))
    Λs = reshape(v[χ^2+1:end], (χ, d))
    return MultiBosonCMPSData_diag{T}(Q, Λs)
end

# operations on the data. not on the cMPS
Base.:+(ψ::MultiBosonCMPSData_diag, ϕ::MultiBosonCMPSData_diag) = MultiBosonCMPSData_diag(ψ.Q + ϕ.Q, ψ.Λs + ϕ.Λs)
Base.:-(ψ::MultiBosonCMPSData_diag, ϕ::MultiBosonCMPSData_diag) = MultiBosonCMPSData_diag(ψ.Q - ϕ.Q, ψ.Λs - ϕ.Λs)
Base.:*(ψ::MultiBosonCMPSData_diag, x::Number) = MultiBosonCMPSData_diag(ψ.Q * x, ψ.Λs * x)
Base.:*(x::Number, ψ::MultiBosonCMPSData_diag) = MultiBosonCMPSData_diag(ψ.Q * x, ψ.Λs * x)
Base.eltype(ψ::MultiBosonCMPSData_diag) = eltype(ψ.Q)
LinearAlgebra.dot(ψ1::MultiBosonCMPSData_diag, ψ2::MultiBosonCMPSData_diag) = dot(ψ1.Q, ψ2.Q) + dot(ψ1.Λs, ψ2.Λs)
TensorKit.inner(ψ1::MultiBosonCMPSData_diag, ψ2::MultiBosonCMPSData_diag) = real(dot(ψ1, ψ2))
LinearAlgebra.norm(ψ::MultiBosonCMPSData_diag) = sqrt(norm(dot(ψ, ψ)))
Base.vec(ψ::MultiBosonCMPSData_diag) = [vec(ψ.Q); vec(ψ.Λs)]

function LinearAlgebra.mul!(w::MultiBosonCMPSData_diag, v::MultiBosonCMPSData_diag, α)
    mul!(w.Q, v.Q, α)
    mul!(w.Λs, v.Λs, α)
    return w
end
function LinearAlgebra.rmul!(v::MultiBosonCMPSData_diag, α)
    rmul!(v.Q, α)
    rmul!(v.Λs, α)
    return v
end
function LinearAlgebra.axpy!(α, ψ1::MultiBosonCMPSData_diag, ψ2::MultiBosonCMPSData_diag)
    axpy!(α, ψ1.Q, ψ2.Q)
    axpy!(α, ψ1.Λs, ψ2.Λs)
    return ψ2
end
function LinearAlgebra.axpby!(α, ψ1::MultiBosonCMPSData_diag, β, ψ2::MultiBosonCMPSData_diag)
    axpby!(α, ψ1.Q, β, ψ2.Q)
    axpby!(α, ψ1.Λs, β, ψ2.Λs)
    return ψ2
end
function Base.similar(ψ::MultiBosonCMPSData_diag) 
    Q = similar(ψ.Q)
    Λs = similar(ψ.Λs)
    return MultiBosonCMPSData_diag(Q, Λs)
end
function randomize!(ψ::MultiBosonCMPSData_diag)
    T = eltype(ψ)
    map!(x -> randn(T), ψ.Q, ψ.Q)
    map!(x -> randn(T), ψ.Λs, ψ.Λs)
end
function Base.zero(ψ::MultiBosonCMPSData_diag) 
    Q = zero(ψ.Q)
    Λs = zero(ψ.Λs)
    return MultiBosonCMPSData_diag(Q, Λs)
end

@inline get_χ(ψ::MultiBosonCMPSData_diag) = size(ψ.Q, 1)
@inline get_d(ψ::MultiBosonCMPSData_diag) = size(ψ.Λs, 2)
#TensorKit.space(ψ::MultiBosonCMPSData_diag) = ℂ^(get_χ(ψ))

function CMPSData(ψ::MultiBosonCMPSData_diag)
    χ, d = get_χ(ψ), get_d(ψ)
    
    Q = TensorMap(ψ.Q, ℂ^χ, ℂ^χ)
    Rs = map(1:d) do ix 
        TensorMap(diagm(ψ.Λs[:, ix]), ℂ^χ, ℂ^χ)
    end
    return CMPSData(Q, Rs)
end

# assuming the R matrices of the input CMPSData are diagonal
function MultiBosonCMPSData_diag(ψ::CMPSData)
    χ, d = get_χ(ψ), get_d(ψ)

    R1 = ψ.Rs[1]
    _, V = eigen(R1)
    Vinv = inv(V)

    Q = convert(Array, Vinv * ψ.Q * V)
    Λs = zeros(eltype(ψ.Q), χ, d)
    for ix in 1:d
        Λs[:, ix] = diag(convert(Array, Vinv * ψ.Rs[ix] * V))
    end
    return MultiBosonCMPSData_diag(Q, Λs)
end

function MultiBosonCMPSData_diag_direct(ψ::CMPSData)
    χ, d = get_χ(ψ), get_d(ψ)

    R1 = ψ.Rs[1]

    Q = convert(Array, ψ.Q)
    Λs = zeros(eltype(ψ.Q), χ, d)
    for ix in 1:d
        Λs[:, ix] = diag(convert(Array, ψ.Rs[ix]))
    end
    return MultiBosonCMPSData_diag(Q, Λs)
end

function right_env(ψ::MultiBosonCMPSData_diag; init = missing, verbosity = 0)
    # transfer matrix
    ψc = CMPSData(ψ)
    fK = TransferMatrix(ψc, ψc)
    
    # solve the fixed-point equation
    vl = right_env(fK; init = init, verbosity = verbosity)
    
    #U, S, _ = svd(convert(Array, vl))
    #return U * Diagonal(S) * U'
    return vl
end
function left_env(ψ::MultiBosonCMPSData_diag; init = missing, verbosity = 0)
    # transfer matrix
    ψc = CMPSData(ψ)
    fK = TransferMatrix(ψc, ψc)
    
    # solve the fixed-point equation
    vl = left_env(fK; init = init, verbosity = verbosity)
    
    #U, S, _ = svd(convert(Array, vl))
    #return U * Diagonal(S) * U'
    return vl
end

function ChainRulesCore.rrule(::Type{CMPSData}, ψ::MultiBosonCMPSData_diag)
    function CMPSData_pushback(∂ψ)
        return NoTangent(), MultiBosonCMPSData_diag_direct(∂ψ)
    end
    return CMPSData(ψ), CMPSData_pushback
end

function expand(ψ::MultiBosonCMPSData_diag, χ::Integer; perturb::Float64=1e-1)
    χ0, d = get_χ(ψ), get_d(ψ)
    if χ <= χ0
        @warn "new χ not bigger than χ0"
        return ψ
    end
    Q = perturb * rand(eltype(ψ), χ, χ)
    Q[1:χ0, 1:χ0] = ψ.Q
    q0 = minimum(real.(eigvals(ψ.Q)))
    Q[diagind(Q)[χ0+1:end]] .+= q0 

    Λs = rand(eltype(ψ), χ, d)
    Λs[1:χ0, 1:d] = ψ.Λs

    return MultiBosonCMPSData_diag(Q, Λs) 
end

function tangent_ovlp_local(ψm::MultiBosonCMPSData_diag, Xm::MultiBosonCMPSData_diag, Ym::MultiBosonCMPSData_diag, EL::MPSBondTensor, ER::MPSBondTensor) where {T,S}
    χ, d = get_χ(ψm), get_d(ψm)
    ψ = CMPSData(ψm)
    X = CMPSData(Xm)
    Y = CMPSData(Ym)
    Id = id(ℂ^χ)
    ER /= tr(EL * ER)

    KY = K_permute(K_otimes(Y.Q, Id) + sum(K_otimes.(Y.Rs, ψ.Rs)))
    KX = K_permute(K_otimes(Id, X.Q) + sum(K_otimes.(ψ.Rs, X.Rs)))
    KYX = K_permute(sum(K_otimes.(Y.Rs, X.Rs)))
    K0 = K_permute(K_otimes(Id, ψ.Q) + K_otimes(ψ.Q, Id) + sum(K_otimes.(ψ.Rs, ψ.Rs)))

    function right_act(K, E)
        @tensor E1[-1; -2] := K[-1 2; 1 -2] * E[1; 2]
        return E1
    end
    function left_act(K, E)
        @tensor E1[-1; -2] := E[1; 2] * K[2 -1; -2 1]
        return E1
    end

    term1 = tr(left_act(KX, EL) * right_act(KY, ER))
    term2 = tr(left_act(KY, EL) * right_act(KX, ER))
    term3 = tr(left_act(KYX, EL) * right_act(K0, ER))
    term4 = tr(left_act(K0, EL) * right_act(KYX, ER))
    return term1 + term2 + term3 + term4
end


function tangent_map_local(ψm::MultiBosonCMPSData_diag, Xm::MultiBosonCMPSData_diag, EL::MPSBondTensor, ER::MPSBondTensor) where {T,S}
    χ, d = get_χ(ψm), get_d(ψm)
    ψ = CMPSData(ψm)
    X = CMPSData(Xm)
    Id = id(ℂ^χ)

    ER /= tr(EL * ER)

    K1 = K_permute(K_otimes(Id, X.Q) + sum(K_otimes.(ψ.Rs, X.Rs)))
    K0 = K_permute(K_otimes(Id, ψ.Q) + K_otimes(ψ.Q, Id) + sum(K_otimes.(ψ.Rs, ψ.Rs)))
    #@tensor ER1[-1; -2] := Kinv[-1 4; 3 -2] * K1[3 2; 1 4] * ER[1; 2]
    #@tensor EL1[-1; -2] := Kinv[4 -1; -2 3] * K1[2 3; 4 1] * EL[1; 2]
    @tensor ER1[-1; -2] := K1[-1 2; 1 -2] * ER[1; 2]
    @tensor EL1[-1; -2] := K1[2 -1; -2 1] * EL[1; 2]
    @tensor ER0[-1; -2] := K0[-1 2; 1 -2] * ER[1; 2]
    @tensor EL0[-1; -2] := K0[2 -1; -2 1] * EL[1; 2]
    @tensor singular = EL[1; 2] * K1[2 3; 4 1] * ER[4; 3]

    mapped_XQ = 1*(EL * ER1 + EL1 * ER) #- singular * EL * ER
    mapped_XRs = map(zip(ψ.Rs, X.Rs)) do (R, XR)
        1*(EL0 * XR * ER + EL * XR * ER0 + EL1 * R * ER + EL * R * ER1) + 1 * EL * XR * ER #- singular * EL * R * ER
    end
    mapped_X_dat = vcat(mapped_XQ.data, [diag(convert(Array, XR)) for XR in mapped_XRs]...)
    return MultiBosonCMPSData_diag(mapped_X_dat, χ, d)

    #mapped_XQ_mat = convert(Array, mapped_XQ)
    #mapped_XRs_mat = zeros(ComplexF64, χ, d)
    #for ix in 1:d
    #    mapped_XRs_mat[:, ix] = diag(convert(Array, mapped_XRs[ix]))
    #end

    #return MultiBosonCMPSData_diag(mapped_XQ_mat, mapped_XRs_mat) 
end




#function tangent_map(ψm::MultiBosonCMPSData_diag, Xm::MultiBosonCMPSData_diag, EL::MPSBondTensor, ER::MPSBondTensor, Kinv::AbstractTensorMap{T, S, 2, 2}) where {T,S}
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
#    return MultiBosonCMPSData_diag(mapped_XQ, mapped_XRs) 
#end

function energy(H::MultiBosonLiebLiniger, ψ::MultiBosonCMPSData_diag; init_envL = missing, init_envR = missing)
    ψn = CMPSData(ψ)
    OH = kinetic(ψn) + H.cs[1,1]* point_interaction(ψn, 1) + H.cs[2,2]* point_interaction(ψn, 2) + H.cs[1,2] * point_interaction(ψn, 1, 2) + H.cs[2,1] * point_interaction(ψn, 2, 1) - H.μs[1] * particle_density(ψn, 1) - H.μs[2] * particle_density(ψn, 2)
    TM = TransferMatrix(ψn, ψn)
    envL = permute(left_env(TM; init = init_envL), (), (1, 2))
    envR = permute(right_env(TM; init = init_envR), (2, 1), ()) 
    return real(tr(envL * OH * envR) / tr(envL * envR))
end 

function ground_state_updateQ(H::MultiBosonLiebLiniger, ψ0::MultiBosonCMPSData_diag; gradtol::Float64=1e-8, do_preconditioning::Bool=false, maxiter::Int=1000, m_LBFGS::Int=8, _finalize! = (x, f, g, numiter) -> (x, f, g, numiter))
    
    cs = Matrix{ComplexF64}(H.cs)
    μs = Vector{ComplexF64}(H.μs)

    function fgE(x::OptimState{Matrix{T}}) where T
        ψ = MultiBosonCMPSData_diag(x.data, ψ0.Λs)
        init_envL, init_envR = x.ρR
        fE_inf = ψ -> energy(H, ψ; init_envL = init_envL, init_envR = init_envR)
        E, ∂ψ = withgradient(fE_inf, ψ)
        return E, ∂ψ[1].Q
    end
    function inner(x::OptimState{Matrix{T}}, a::Matrix, b::Matrix) where T
        # be careful the cases with or without a factor of 2. depends on how to define the complex gradient
        return real(dot(a, b)) 
    end
    function retract(x::OptimState{Matrix{T}}, dQ::Matrix, α::Real) where T
        Q = x.data + α * dQ
        return OptimState(Q, missing, x.ρR, x.prev, x.df), dQ
    end

    function _no_precondition(x::OptimState{Matrix{T}}, dQ::Matrix) where T
        return dQ
    end
    function _precondition(x::OptimState{Matrix{T}}, dQ::Matrix) where T
        ψ = MultiBosonCMPSData_diag(x.data, ψ0.Λs)
        χ, d = get_χ(ψ), get_d(ψ)

        EL = left_env(ψ; init = x.ρR[1])
        ER = right_env(ψ; init = x.ρR[2])
        ER /= tr(EL * ER)

        ELmat = convert(Array, EL)
        ERmat = convert(Array, ER)
        U, S, V = svd(ELmat)
        ELmat = U * Diagonal(S) * U'
        U, S, V = svd(ERmat)
        ERmat = U * Diagonal(S) * U'
        
        δ = isnan(x.df) ? 1e-3 : max(x.df, 1e-12)
        ELmat[diagind(ELmat)] .+= sqrt(δ)
        ERmat[diagind(ERmat)] .+= sqrt(δ)

        dQp =inv(ELmat) * dQ * inv(ERmat)

        return dQp
    end

    function finalize!(x::OptimState{Matrix{T}}, f, g, numiter) where T
        _finalize!(x, f, g, numiter)

        α = norm(f) ^ (1/3)
        x.prev = α # prev now plays the role of α. change name. 

        #g1 = α^(-2.5) * g # it seems that V and W also need to be scaled sperately
        g1 = g
        δ = norm(g1) ^ 2
        x.df = δ # FIXME. df now plays the role of δ. change name.
        
        println("finalize: δ = $(δ), α = $(α), cond number = $(cond(x.data)), eigenvalues = $(abs.(svd(x.data).S)))")
        ψ = MultiBosonCMPSData_diag(x.data, ψ0.Λs)
        x.ρR = (left_env(ψ; init = x.ρR[1], verbosity = 1), right_env(ψ; init = x.ρR[2], verbosity = 1))
        return x, f, g, numiter
    end
    transport!(v, x, d, α, xnew) = v

    optalg_LBFGS = LBFGS(m_LBFGS; maxiter=maxiter, gradtol=gradtol, acceptfirst=false, verbosity=2)

    if do_preconditioning
        @show "doing precondition"
        precondition★ = _precondition
    else
        @show "no precondition"
        precondition★ = _no_precondition
    end

    x0 = OptimState(ψ0.Q) 
    x0.ρR = (left_env(ψ0; init = missing, verbosity = 2), right_env(ψ0; init = missing, verbosity = 2))
    x1, E1, grad1, numfg1, history1 = optimize(fgE, x0, optalg_LBFGS; retract = retract,
                                    precondition = precondition★,
                                    inner = inner, transport! =transport!, finalize! = finalize!
                                    );

    res = (MultiBosonCMPSData_diag(x1.data, ψ0.Λs), E1, grad1, numfg1, history1)
    return res

end

function ground_state(H::MultiBosonLiebLiniger, ψ0::MultiBosonCMPSData_diag; gradtol::Float64=1e-8, do_preconditioning::Bool=false, maxiter::Int=1000, m_LBFGS::Int=8, _finalize! = (x, f, g, numiter) -> (x, f, g, numiter))

    if H.L < Inf
        error("finite size not implemented yet.")
    end

    cs = Matrix{ComplexF64}(H.cs)
    μs = Vector{ComplexF64}(H.μs)

    function fgE(x::OptimState{MultiBosonCMPSData_diag{T}}) where T
        ψ = x.data
        init_envL, init_envR = x.ρR
        fE_inf = ψ -> energy(H, ψ; init_envL = init_envL, init_envR = init_envR)
        E, ∂ψ = withgradient(fE_inf, ψ)
        return E, ∂ψ[1]
    end
    function inner(x::OptimState{MultiBosonCMPSData_diag{T}}, a::MultiBosonCMPSData_diag, b::MultiBosonCMPSData_diag) where T
        # be careful the cases with or without a factor of 2. depends on how to define the complex gradient
        return real(dot(a, b)) 
    end
    function retract(x::OptimState{MultiBosonCMPSData_diag{T}}, dψ::MultiBosonCMPSData_diag, α::Real) where T
        ψ = x.data
        Λs = ψ.Λs .+ α .* dψ.Λs 
        Q = ψ.Q + α * dψ.Q
        ψ1 = MultiBosonCMPSData_diag(Q, Λs)
        return OptimState(ψ1, missing, x.ρR, x.prev, x.df), dψ
    end
    function scale!(dψ::MultiBosonCMPSData_diag, α::Number)
        dψ.Q .= dψ.Q * α
        dψ.Λs .= dψ.Λs .* α
        return dψ
    end
    function add!(dψ::MultiBosonCMPSData_diag, dψ1::MultiBosonCMPSData_diag, α::Number=1, β::Number=1) 
        dψ.Q .+= dψ1.Q * α
        dψ.Λs .+= dψ1.Λs .* α
        return dψ
    end

    function _no_precondition(x::OptimState{MultiBosonCMPSData_diag{T}}, dψ::MultiBosonCMPSData_diag) where T
        return dψ
    end
    function _precondition(x::OptimState{MultiBosonCMPSData_diag{T}}, dψ::MultiBosonCMPSData_diag) where T
        # TODO. avoid the re-computation of K, λ, EL, ER, Kinv
        # FIXME. this function is already broken
        ψn = CMPSData(ψ0)
        K = K_permute(K_mat(ψn, ψn))
        λ, EL = left_env(K)
        λ, ER = right_env(K)
        Kinv = Kmat_pseudo_inv(K, λ)

        ϵ = max(1e-12, 1e-3*norm(dψ))
        mapped, _ = linsolve(X -> tangent_map(ψ0, X, EL, ER, Kinv) + ϵ*X, dψ, dψ; maxiter=250, ishermitian = true, isposdef = true, tol=ϵ)
        return mapped
    end
    function _precondition_local(x::OptimState{MultiBosonCMPSData_diag{T}}, dψ::MultiBosonCMPSData_diag) where T 
        ψ = x.data
        χ, d = get_χ(ψ), get_d(ψ)

        EL = left_env(ψ; init = x.ρR[1])
        ER = right_env(ψ; init = x.ρR[2])

        #for ix in 1:(χ^2+d*χ)
        #    for iy in 1:(χ^2+d*χ)
        #        vx = zeros(ComplexF64, χ^2+d*χ)
        #        vx[ix] = 1
        #        gx = MultiBosonCMPSData_diag(vx, χ, d)
        #        vy = zeros(ComplexF64, χ^2+d*χ)
        #        vy[iy] = 1
        #        gy = MultiBosonCMPSData_diag(vy, χ, d)

        #        ovlp1 = tangent_ovlp_local(ψ, gx, gy, EL, ER)
        #        ovlp2 = dot(gy, tangent_map_local(ψ, gx, EL, ER))
        #        if abs(ovlp1 - ovlp2) > 1e-6
        #            @show ix, iy, ovlp1, ovlp2
        #            error("error happened!")
        #        end
        #    end
        #end

        if ismissing(x.preconditioner)
            δ = isnan(x.df) ? 1e-3 : max(x.df, 1e-12)
            α = isnan(x.prev) ? 1.0 : x.prev

            P = zeros(ComplexF64, χ^2+d*χ, χ^2+d*χ)

            blas_num_threads = LinearAlgebra.BLAS.get_num_threads()
            LinearAlgebra.BLAS.set_num_threads(1)

            Threads.@threads for ix in 1:(χ^2+d*χ)
                v = zeros(ComplexF64, χ^2+d*χ)
                v[ix] = 1
                g = MultiBosonCMPSData_diag(v, χ, d)
                g1 = tangent_map_local(ψ, g, EL, ER)
                P[:, ix] = vec(g1)
            end 
            LinearAlgebra.BLAS.set_num_threads(blas_num_threads)
            #@show norm(P - P')
            Peigs = eigvals(P)
            P[diagind(P)] .-= minimum(Peigs[1])#@show real(Peigs[1]), real(Peigs[χ+1]), real(Peigs[end])
                        
            P[diagind(P)[1:d*χ]] .+= δ # dD
            P[diagind(P)[d*χ+1:end]] .+= δ #* α # X
            
            Peigs = eigvals(P)
            #@show real(Peigs[1]), real(Peigs[χ+1]), real(Peigs[end])

            x.preconditioner = qr(P)
        end
        vp = x.preconditioner \ vec(dψ)
        PG = MultiBosonCMPSData_diag(vp, χ, d)
        return PG

        #ψ = x.data
        #EL, ER = x.ρR

        #δ = isnan(x.df) ? 1e-3 : max(x.df, 1e-12)
        #v_mapped, _ = linsolve(X -> tangent_map_local(ψ, X, EL, ER) + δ*X, dψ, dψ; maxiter=1, ishermitian = true, isposdef = true, tol= 0.0, verbosity = 0)

        #return v_mapped
    end

    function finalize!(x::OptimState{MultiBosonCMPSData_diag{T}}, f, g, numiter) where T
        _finalize!(x, f, g, numiter)

        α = norm(f) ^ (1/3)
        x.prev = α # prev now plays the role of α. change name. 

        #g1 = α^(-2.5) * g # it seems that V and W also need to be scaled sperately
        g1 = g
        δ = norm(g1) ^ 2
        x.df = δ # FIXME. df now plays the role of δ. change name.
        
        x.ρR = (left_env(x.data; init = x.ρR[1], verbosity = 1), right_env(x.data; init = x.ρR[2], verbosity = 1))

        full_env = convert(Array, x.ρR[1] * x.ρR[2])

        println("finalize: δ = $(δ), α = $(α), cond number = $(cond(full_env))")

        return x, f, g, numiter
    end
    transport!(v, x, d, α, xnew) = v

    optalg_LBFGS = LBFGS(m_LBFGS; maxiter=maxiter, gradtol=gradtol, acceptfirst=false, verbosity=2)

    if do_preconditioning
        @show "doing precondition"
        precondition★ = _precondition_local
    else
        @show "no precondition"
        precondition★ = _no_precondition
    end

    x0 = OptimState(ψ0) 
    x0.ρR = (left_env(x0.data; init = missing, verbosity = 2), right_env(x0.data; init = missing, verbosity = 2))
    x1, E1, grad1, numfg1, history1 = optimize(fgE, x0, optalg_LBFGS; retract = retract,
                                    precondition = precondition★,
                                    inner = inner, transport! =transport!,
                                    scale! = scale!, add! = add!, finalize! = finalize!
                                    );

    res = (x1.data, E1, grad1, numfg1, history1)
    return res
end

function VectorInterface.scalartype(a::MultiBosonCMPSData_diag)
    return eltype(a.Q)
end
function VectorInterface.zerovector(a::MultiBosonCMPSData_diag)
    return MultiBosonCMPSData_diag(zerovector(a.Q), zerovector(a.Λs))
end
function VectorInterface.zerovector!(a::MultiBosonCMPSData_diag)
    zerovector!(a.Q)
    zerovector!(a.Λs)
    return a
end
function VectorInterface.zerovector!!(a::MultiBosonCMPSData_diag)
    zerovector!!(a.Q)
    zerovector!!(a.Λs)
    return a
end
function VectorInterface.scale(a::MultiBosonCMPSData_diag, α::Number)
    Q = scale(a.Q, α)
    Λs = scale(a.Λs, α)
    return MultiBosonCMPSData_diag(Q, Λs)
end
function VectorInterface.scale!(a::MultiBosonCMPSData_diag, α::Number)
    scale!(a.Q, α)
    scale!(a.Λs, α)
    return a
end
function VectorInterface.scale!!(a::MultiBosonCMPSData_diag, α::Number)
    scale!!(a.Q, α)
    scale!!(a.Λs, α)
    return a
end
function VectorInterface.add(a::MultiBosonCMPSData_diag, b::MultiBosonCMPSData_diag, α::Number=1, β::Number=1)
    Q = VectorInterface.add(a.Q, b.Q, α, β)
    Λs = VectorInterface.add(a.Λs, b.Λs, α, β)
    return MultiBosonCMPSData_diag(Q, Λs)
end
function VectorInterface.add!(a::MultiBosonCMPSData_diag, b::MultiBosonCMPSData_diag, α::Number=1, β::Number=1)
    VectorInterface.add!(a.Q, b.Q, α, β)
    VectorInterface.add!(a.Λs, b.Λs, α, β)
    return a
end
function VectorInterface.add!!(a::MultiBosonCMPSData_diag, b::MultiBosonCMPSData_diag, α::Number=1, β::Number=1)
    VectorInterface.add!!(a.Q, b.Q, α, β)
    VectorInterface.add!!(a.Λs, b.Λs, α, β)
    return a
end
Base.similar(a::MultiBosonCMPSData_diag) = MultiBosonCMPSData_diag(similar(a.Q), similar(a.Λs))
Base.vec(a::MultiBosonCMPSData_diag) = vcat(vec(a.Q), vec(a.Λs))