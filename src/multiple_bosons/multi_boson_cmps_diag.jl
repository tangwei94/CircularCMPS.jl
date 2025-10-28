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
    Q = 0.1 * randn(eltype(ψ), χ, χ)
    Q[1:χ0, 1:χ0] = ψ.Q

    Λs = perturb * randn(eltype(ψ), χ, d)
    Λs[1:χ0, 1:d] = ψ.Λs

    return MultiBosonCMPSData_diag(Q, Λs) 
end

function tangent_map_local(ψm::MultiBosonCMPSData_diag, Xm::MultiBosonCMPSData_diag, EL::MPSBondTensor, ER::MPSBondTensor) where {T,S}
    χ = get_χ(ψm)
    ψ = CMPSData(ψm)
    X = CMPSData(Xm)
    Id = id(ℂ^χ)

    ER /= tr(EL * ER)

    K1 = K_permute(K_otimes(Id, X.Q) + sum(K_otimes.(ψ.Rs, X.Rs)))
    #@tensor ER1[-1; -2] := Kinv[-1 4; 3 -2] * K1[3 2; 1 4] * ER[1; 2]
    #@tensor EL1[-1; -2] := Kinv[4 -1; -2 3] * K1[2 3; 4 1] * EL[1; 2]
    @tensor ER1[-1; -2] := K1[-1 2; 1 -4] * ER[1; 2]
    @tensor EL1[-1; -2] := K1[2 -1; -2 1] * EL[1; 2]
    @tensor singular = EL[1; 2] * K1[2 3; 4 1] * ER[4; 3]

    mapped_XQ = EL * ER1 + EL1 * ER + singular * EL * ER
    mapped_XRs = map(zip(ψ.Rs, X.Rs)) do (R, XR)
        EL * XR * ER + EL1 * R * ER + EL * R * ER1 + singular * EL * R * ER
    end

    return MultiBosonCMPSData_diag(mapped_XQ, mapped_XRs) 
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

function ground_state(H::MultiBosonLiebLiniger, ψ0::MultiBosonCMPSData_diag; gradtol::Float64=1e-8, do_preconditioning::Bool=false, maxiter::Int=1000, _finalize! = (x, f, g, numiter) -> (x, f, g, numiter))
    if H.L == Inf
        cs = Matrix{ComplexF64}(H.cs)
        μs = Vector{ComplexF64}(H.μs)

        function fE_inf(ψ::MultiBosonCMPSData_diag)
            ψn = CMPSData(ψ)
            OH = kinetic(ψn) + H.cs[1,1]* point_interaction(ψn, 1) + H.cs[2,2]* point_interaction(ψn, 2) + H.cs[1,2] * point_interaction(ψn, 1, 2) + H.cs[2,1] * point_interaction(ψn, 2, 1) - H.μs[1] * particle_density(ψn, 1) - H.μs[2] * particle_density(ψn, 2)
            TM = TransferMatrix(ψn, ψn)
            envL = permute(left_env(TM), (), (1, 2))
            envR = permute(right_env(TM), (2, 1), ()) 
            return real(tr(envL * OH * envR) / tr(envL * envR))
        end
        #function fE_inf(ψm::MultiBosonCMPSData_diag)
        #    ψ = CMPSData(ψm)
        #    OH = kinetic(ψ) + point_interaction(ψ, cs) - particle_density(ψ, μs)

        #    TM = TransferMatrix(ψ, ψ)
        #    envL = permute(left_env(TM), (), (1, 2))
        #    envR = permute(right_env(TM), (2, 1), ()) 
        #    return real(tr(envL * OH * envR) / tr(envL * envR))
        #end
        @show "infinite system"
    
        function fgE(ψ::MultiBosonCMPSData_diag)
            E = fE_inf(ψ)
            ∂ψ = fE_inf'(ψ) 
            return E, ∂ψ 
        end
    
        function inner(ψ, ψ1::MultiBosonCMPSData_diag, ψ2::MultiBosonCMPSData_diag)
            # be careful the cases with or without a factor of 2. depends on how to define the complex gradient
            return real(dot(ψ1, ψ2)) 
        end

        function retract(ψ::MultiBosonCMPSData_diag, dψ::MultiBosonCMPSData_diag, α::Real)
            Λs = ψ.Λs .+ α .* dψ.Λs 
            Q = ψ.Q + α * dψ.Q
            ψ1 = MultiBosonCMPSData_diag(Q, Λs)
            return ψ1, dψ
        end

        function scale!(dψ::MultiBosonCMPSData_diag, α::Number)
            dψ.Q .= dψ.Q * α
            dψ.Λs .= dψ.Λs .* α
            return dψ
        end

        function add!(dψ::MultiBosonCMPSData_diag, dψ1::MultiBosonCMPSData_diag, α::Number) 
            dψ.Q .+= dψ1.Q * α
            dψ.Λs .+= dψ1.Λs .* α
            return dψ
        end

        # only for comparison
        function _no_precondition(ψ::MultiBosonCMPSData_diag, dψ::MultiBosonCMPSData_diag)
            return dψ
        end

        function _precondition(ψ0::MultiBosonCMPSData_diag, dψ::MultiBosonCMPSData_diag)
            # TODO. avoid the re-computation of K, λ, EL, ER, Kinv
            ψn = CMPSData(ψ0)
            K = K_permute(K_mat(ψn, ψn))
            λ, EL = left_env(K)
            λ, ER = right_env(K)
            Kinv = Kmat_pseudo_inv(K, λ)

            ϵ = max(1e-12, 1e-3*norm(dψ))
            mapped, _ = linsolve(X -> tangent_map(ψ0, X, EL, ER, Kinv) + ϵ*X, dψ, dψ; maxiter=250, ishermitian = true, isposdef = true, tol=ϵ)
            return mapped

            #χ, d = get_χ(ψ0), get_d(ψ0)
            #M = zeros(ComplexF64, χ^2+d*χ, χ^2+d*χ)
            #for ix in 1:(χ^2+d*χ)
            #    v = zeros(χ^2+d*χ)
            #    v[ix] = 1
          
            #    X = MultiBosonCMPSData(v, χ, d)
            #    v1 = vec(tangent_map(ψ0, X, EL, ER, Kinv))
            #    M[:, ix] = v1
            #end

            #λs, V = eigen(Hermitian(M))
            #λs[1] < -1e-9 && @warn "$(λs[1]) not positive definite"
            
            #dV = vec(dψ)
            #mappedV = V'[:, χ:end] * Diagonal(1 ./ (λs[χ:end] .+ ϵ)) * V[χ:end, :] * dV
            #return MultiBosonCMPSData(mappedV, χ, d)
        end

        transport!(v, x, d, α, xnew) = v

        optalg_LBFGS = LBFGS(;maxiter=maxiter, gradtol=gradtol, verbosity=2)

        if do_preconditioning
            @show "doing precondition"
            precondition = _precondition
        else
            @show "no precondition"
            precondition = _no_precondition
        end
        ψ1, E1, grad1, numfg1, history1 = optimize(fgE, ψ0, optalg_LBFGS; retract = retract,
                                        precondition = precondition,
                                        inner = inner, transport! =transport!,
                                        scale! = scale!, add! = add!, finalize! = _finalize!
                                        );

        res1 = (ψ1, E1, grad1, numfg1, history1)
        return res1

    else
        @show "finite system of size $(H.L)"

        error("finite size not implemented yet.")
    end
end