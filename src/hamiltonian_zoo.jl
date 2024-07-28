abstract type AbstractHamiltonian end

struct SingleBosonLiebLiniger <: AbstractHamiltonian
    c::Real
    μ::Real
    L::Real
end

function ground_state(H::SingleBosonLiebLiniger, ψ0::CMPSData)
    if H.L == Inf
        function fE_inf(ψ::CMPSData)
            OH = kinetic(ψ) + H.c*point_interaction(ψ) - H.μ * particle_density(ψ)
            TM = TransferMatrix(ψ, ψ)
            envL = permute(left_env(TM), (), (1, 2))
            envR = permute(right_env(TM), (2, 1), ()) 
            return real(tr(envL * OH * envR) / tr(envL * envR))
        end
        @show "infinite system"

        return minimize(fE_inf, ψ0, CircularCMPSRiemannian(1000, 1e-9, 2)) # TODO. change this as input. 
    else
        @show "finite system of size $(H.L)"
        function fE_finiteL(ψ::CMPSData)
            OH = kinetic(ψ) + H.c*point_interaction(ψ) - H.μ * particle_density(ψ)
            expK, _ = finite_env(K_mat(ψ, ψ), H.L)
            return real(tr(expK * OH))
        end 

        return minimize(fE_finiteL, ψ0, CircularCMPSRiemannian(1000, 1e-9, 2)) # TODO. change this as input. 
    end
end

struct MultiBosonLiebLiniger <: AbstractHamiltonian
    cs::Matrix{<:Real}
    μs::Vector{<:Real}
    L::Real
end

#struct MultiBosonCMPSState
#    ψ::MultiBosonCMPSData
#    envL::MPSBondTensor
#    envR::MPSBondTensor
#    λ::Number
#    K::AbstractTensorMap
#    Kinv::AbstractTensorMap
#end

#function MultiBosonCMPSState(ψ::MultiBosonCMPSData)
#    ψn = CMPSData(ψ0)
#    K = K_permute(K_mat(ψn, ψn))
#    λ, EL = left_env(K)
#    λ, ER = right_env(K)
#    Kinv = Kmat_pseudo_inv(K, λ)
#    return MultiBosonCMPSState(ψ, EL, ER, λ, K, Kinv)
#end

function ground_state(H::MultiBosonLiebLiniger, ψ0::MultiBosonCMPSData; do_preconditioning::Bool=true, maxiter::Int=10000)
    if H.L == Inf
        cs = Matrix{ComplexF64}(H.cs)
        μs = Vector{ComplexF64}(H.μs)

        function fE_inf(ψ::MultiBosonCMPSData)
            ψn = CMPSData(ψ)
            OH = kinetic(ψn) + H.cs[1,1]* point_interaction(ψn, 1) + H.cs[2,2]* point_interaction(ψn, 2) + H.cs[1,2] * point_interaction(ψn, 1, 2) + H.cs[2,1] * point_interaction(ψn, 2, 1) - H.μs[1] * particle_density(ψn, 1) - H.μs[2] * particle_density(ψn, 2)
            TM = TransferMatrix(ψn, ψn)
            envL = permute(left_env(TM), (), (1, 2))
            envR = permute(right_env(TM), (2, 1), ()) 
            return real(tr(envL * OH * envR) / tr(envL * envR))
        end
        #function fE_inf(ψm::MultiBosonCMPSData)
        #    ψ = CMPSData(ψm)
        #    OH = kinetic(ψ) + point_interaction(ψ, cs) - particle_density(ψ, μs)

        #    TM = TransferMatrix(ψ, ψ)
        #    envL = permute(left_env(TM), (), (1, 2))
        #    envR = permute(right_env(TM), (2, 1), ()) 
        #    return real(tr(envL * OH * envR) / tr(envL * envR))
        #end
        @show "infinite system"
    
        function fgE(ψ::MultiBosonCMPSData)
            E = fE_inf(ψ)
            ∂ψ = fE_inf'(ψ) 
            return E, ∂ψ 
        end
    
        function inner(ψ, ψ1::MultiBosonCMPSData, ψ2::MultiBosonCMPSData)
            # be careful the cases with or without a factor of 2. depends on how to define the complex gradient
            return real(dot(ψ1, ψ2)) 
        end

        function retract(ψ::MultiBosonCMPSData, dψ::MultiBosonCMPSData, α::Real)
            Λs = ψ.Λs .+ α .* dψ.Λs 
            Q = ψ.Q + α * dψ.Q
            ψ1 = MultiBosonCMPSData(Q, Λs)
            return ψ1, dψ
        end

        function scale!(dψ::MultiBosonCMPSData, α::Number)
            dψ.Q = dψ.Q * α
            dψ.Λs .= dψ.Λs .* α
            return dψ
        end

        function add!(dψ::MultiBosonCMPSData, dψ1::MultiBosonCMPSData, α::Number) 
            dψ.Q += dψ1.Q * α
            dψ.Λs .+= dψ1.Λs .* α
            return dψ
        end

        # only for comparison
        function _no_precondition(ψ::MultiBosonCMPSData, dψ::MultiBosonCMPSData)
            return dψ
        end

        function _precondition(ψ0::MultiBosonCMPSData, dψ::MultiBosonCMPSData)
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

        optalg_LBFGS = LBFGS(;maxiter=maxiter, gradtol=1e-8, verbosity=2)

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
                                        scale! = scale!, add! = add!
                                        );

        res1 = (ψ1, E1, grad1, numfg1, history1)
        return res1

    else
        @show "finite system of size $(H.L)"

        error("finite size not implemented yet.")
    end
end

function ground_state(H::MultiBosonLiebLiniger, ψ0::MultiBosonCMPSData_P; do_preconditioning::Bool=true, maxiter::Int=10000)
    if H.L == Inf
        cs = Matrix{ComplexF64}(H.cs)
        μs = Vector{ComplexF64}(H.μs)

        function fE_inf(ψ::MultiBosonCMPSData_P)
            ψn = CMPSData(ψ)
            OH = kinetic(ψn) + H.cs[1,1]* point_interaction(ψn, 1) + H.cs[2,2]* point_interaction(ψn, 2) + H.cs[1,2] * point_interaction(ψn, 1, 2) + H.cs[2,1] * point_interaction(ψn, 2, 1) - H.μs[1] * particle_density(ψn, 1) - H.μs[2] * particle_density(ψn, 2)
            TM = TransferMatrix(ψn, ψn)
            envL = permute(left_env(TM), (), (1, 2))
            envR = permute(right_env(TM), (2, 1), ()) 
            return real(tr(envL * OH * envR) / tr(envL * envR))
        end
        #function fE_inf(ψm::MultiBosonCMPSData)
        #    ψ = CMPSData(ψm)
        #    OH = kinetic(ψ) + point_interaction(ψ, cs) - particle_density(ψ, μs)

        #    TM = TransferMatrix(ψ, ψ)
        #    envL = permute(left_env(TM), (), (1, 2))
        #    envR = permute(right_env(TM), (2, 1), ()) 
        #    return real(tr(envL * OH * envR) / tr(envL * envR))
        #end
        @show "infinite system"
    
        function fgE(ψ::MultiBosonCMPSData_P)
            E = fE_inf(ψ)
            ∂ψ = fE_inf'(ψ) 
            return E, ∂ψ 
        end
    
        function inner(ψ, ψ1::MultiBosonCMPSData_P, ψ2::MultiBosonCMPSData_P)
            # be careful the cases with or without a factor of 2. depends on how to define the complex gradient
            return real(dot(ψ1, ψ2)) 
        end

        function retract(ψ::MultiBosonCMPSData_P, dψ::MultiBosonCMPSData_P, α::Real)
            Ms = ψ.Ms .+ α .* dψ.Ms 
            Q = ψ.Q + α * dψ.Q
            ψ1 = MultiBosonCMPSData_P(Q, Ms)
            return ψ1, dψ
        end

        function scale!(dψ::MultiBosonCMPSData_P, α::Number)
            dψ.Q = dψ.Q * α
            dψ.Ms .= dψ.Ms .* α
            return dψ
        end

        function add!(dψ::MultiBosonCMPSData_P, dψ1::MultiBosonCMPSData_P, α::Number) 
            dψ.Q += dψ1.Q * α
            dψ.Ms .+= dψ1.Ms .* α
            return dψ
        end

        # only for comparison
        function _no_precondition(ψ::MultiBosonCMPSData_P, dψ::MultiBosonCMPSData_P)
            return dψ
        end

        function _precondition(ψ0::MultiBosonCMPSData_P, dψ::MultiBosonCMPSData_P)
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

        optalg_LBFGS = LBFGS(;maxiter=maxiter, gradtol=1e-8, verbosity=2)

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
                                        scale! = scale!, add! = add!
                                        );

        res1 = (ψ1, E1, grad1, numfg1, history1)
        return res1

    else
        @show "finite system of size $(H.L)"

        error("finite size not implemented yet.")
    end
end

"""
    ground_state(H::MultiBosonLiebLiniger, ψ0::CMPSData; Λ::Real=1.0)

Find the ground state of the MultiBosonLiebLiniger model with the given Hamiltonian `H` and the initial state `ψ0`. 
The multi-boson cMPS is parametrized with no regularity conditions. This allows a more efficient optimization. 
The regularity condition is achieved via an additional Lagrangian multiplier term `Λ [ψ1, ψ2]† [ψ1, ψ2]`.
"""
function ground_state(H::MultiBosonLiebLiniger, ψ0::CMPSData; Λs::Vector{<:Real}=sqrt(10) .^ (4:10), gradtol=1e-4, maxiter=1000, do_benchmark=true, fϵ=(x->1e-3*x))
    function fE_inf(ψ::CMPSData, Λ::Real) 
        OH = kinetic(ψ) + H.cs[1,1]* point_interaction(ψ, 1) + H.cs[2,2]* point_interaction(ψ, 2) + H.cs[1,2] * point_interaction(ψ, 1, 2) + H.cs[2,1] * point_interaction(ψ, 2, 1) - H.μs[1] * particle_density(ψ, 1) - H.μs[2] * particle_density(ψ, 2) + lagrangian_multiplier(ψ, 1, 2, Λ) 
        TM = TransferMatrix(ψ, ψ)
        envL = permute(left_env(TM), (), (1, 2))
        envR = permute(right_env(TM), (2, 1), ()) 
        return real(tr(envL * OH * envR) / tr(envL * envR))
    end
    function fE_finiteL(ψ::CMPSData, Λ::Real)
        OH = kinetic(ψ) + H.cs[1,1]* point_interaction(ψ, 1) + H.cs[2,2]* point_interaction(ψ, 2) + H.cs[1,2] * point_interaction(ψ, 1, 2) + H.cs[2,1] * point_interaction(ψ, 2, 1) - H.μs[1] * particle_density(ψ, 1) - H.μs[2] * particle_density(ψ, 2) + lagrangian_multiplier(ψ, 1, 2, Λ) 
        expK, _ = finite_env(K_mat(ψ, ψ), H.L)
        return real(tr(expK * OH))
    end 
    function _finalize!(x, f, g, numiter)
        x1, err = convert_to_MultiBosonCMPSData_MDMinv(x.data)
        x1 = left_canonical(x1)
        fMDMinv = (xm -> fE(CMPSData(xm), 0))
        f1, diff1 = withgradient(fMDMinv, x1)
        g1 = diff_to_grad(x1, diff1[1])
        push!(E_history, f1)
        push!(gnorm_history, norm(g1))
        push!(err_history, err)
        println("iter $numiter: E: $f gnorm: $(norm(g)) E1: $f1 g1norm: $(norm(g1)) err: $err")

        if f1 < optimal_E
            optimal_solution = deepcopy(x1)
            optimal_E = f1
            optimal_grad = deepcopy(g1)
        end

        return x, f, g, numiter
    end

    fE = (H.L == Inf) ? fE_inf : fE_finiteL
    finalize! = do_benchmark ? _finalize! : OptimKit._finalize!

    ψ = left_canonical(ψ0)[2]
    E, grad = withgradient(x->fE_inf(x, Λs[1]), ψ)
    grad = grad[1]
    total_numfg = 0
    E_history, gnorm_history, err_history = Float64[], Float64[], Float64[]

    optimal_solution = convert_to_MultiBosonCMPSData_MDMinv(ψ)[1]
    optimal_E = Inf
    optimal_grad = deepcopy(grad)

    for Λ in Λs
        ψ, E, grad, numfg, history = minimize(x->fE(x, Λ), ψ, CircularCMPSRiemannian(maxiter, gradtol, 1); finalize! = finalize!, fϵ=fϵ)
        total_numfg += numfg
    end
    return optimal_solution, optimal_E, optimal_grad, total_numfg, hcat(E_history, gnorm_history, err_history)
end

function ground_state(H::MultiBosonLiebLiniger, ψ0::MultiBosonCMPSData_MDMinv; do_preconditioning::Bool=true, maxiter::Int=10000, gradtol=1e-6, fϵ=identity)
    if H.L < Inf
        error("finite size not implemented yet.")
    end

    function fE_inf(ψ::MultiBosonCMPSData_MDMinv)
        ψn = CMPSData(ψ)
        OH = kinetic(ψn) + H.cs[1,1]* point_interaction(ψn, 1) + H.cs[2,2]* point_interaction(ψn, 2) + H.cs[1,2] * point_interaction(ψn, 1, 2) + H.cs[2,1] * point_interaction(ψn, 2, 1) - H.μs[1] * particle_density(ψn, 1) - H.μs[2] * particle_density(ψn, 2)
        TM = TransferMatrix(ψn, ψn)
        envL = permute(left_env(TM), (), (1, 2))
        envR = permute(right_env(TM), (2, 1), ()) 
        return real(tr(envL * OH * envR) / tr(envL * envR))
    end
    
    function fgE(x::OptimState{MultiBosonCMPSData_MDMinv{T}}) where T
        ψ = x.data
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
        return OptimState(ψ1, missing, x.prev, x.df), dψ
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

    # linsolve is slow
    #function _precondition_linsolve(ψ0::MultiBosonCMPSData_MDMinv, dψ::MultiBosonCMPSData_MDMinv_Grad)
    #    ϵ = max(1e-12, min(1, norm(dψ)^1.5))
    #    χ, d = get_χ(ψ0), get_d(ψ0)
    #    function f_map(v)
    #        g = MultiBosonCMPSData_MDMinv_Grad(v, χ, d)
    #        return vec(tangent_map(ψ0, g)) + ϵ*v
    #    end

    #    vp, _ = linsolve(f_map, vec(dψ), rand(ComplexF64, χ*d+χ^2); maxiter=1000, ishermitian = true, isposdef = true, tol=ϵ)
    #    return MultiBosonCMPSData_MDMinv_Grad(vp, χ, d)
    #end
    function _precondition(x::OptimState{MultiBosonCMPSData_MDMinv{T}}, dψ::MultiBosonCMPSData_MDMinv_Grad) where T
        ψ = x.data
        χ, d = get_χ(ψ), get_d(ψ)

        if ismissing(x.preconditioner)
            ϵ = isnan(x.df) ? 1e-3*fϵ(norm(dψ)) : fϵ(x.df)
            #ϵ = 1e-3*fϵ(norm(dψ)) 
            ϵ = max(1e-12, ϵ)

            P = zeros(ComplexF64, χ^2+d*χ, χ^2+d*χ)

            blas_num_threads = LinearAlgebra.BLAS.get_num_threads()
            LinearAlgebra.BLAS.set_num_threads(1)
            Threads.@threads for ix in 1:(χ^2+d*χ)
                v = zeros(ComplexF64, χ^2+d*χ)
                v[ix] = 1
                g = MultiBosonCMPSData_MDMinv_Grad(v, χ, d)
                g1 = tangent_map(ψ, g)
                P[:, ix] = vec(g1)
            end 
            LinearAlgebra.BLAS.set_num_threads(blas_num_threads)
            P[diagind(P)] .+= ϵ
            x.preconditioner = P
        end
        vp = x.preconditioner \ vec(dψ)
        return MultiBosonCMPSData_MDMinv_Grad(vp, χ, d)
    end

    transport!(v, x, d, α, xnew) = v

    function finalize!(x::OptimState{MultiBosonCMPSData_MDMinv{T}}, f, g, numiter) where T
        x.preconditioner = missing
        x.df = abs(f - x.prev)
        x.prev = f
        return x, f, g, numiter
    end

    optalg_LBFGS = LBFGS(;maxiter=maxiter, gradtol=gradtol, verbosity=2)

    if do_preconditioning
        @show "doing precondition"
        precondition = _precondition
    else
        @show "no precondition"
        precondition = _no_precondition
    end

    x0 = OptimState(left_canonical(ψ0)) # FIXME. needs to do it twice??
    x1, E1, grad1, numfg1, history1 = optimize(fgE, x0, optalg_LBFGS; retract = retract,
                                    precondition = precondition,
                                    inner = inner, transport! =transport!,
                                    scale! = scale!, add! = add!, finalize! = finalize!
                                    );

    res = (x1.data, E1, grad1, numfg1, history1)
    return res
end