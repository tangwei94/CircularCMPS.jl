mutable struct OptimState{A}
    data::A
    preconditioner::Union{Matrix{ComplexF64}, Nothing}
    prev::Float64
    df::Float64

    function OptimState(data::A, preconditioner::Union{Matrix{ComplexF64}, Nothing}, prev::Float64, df::Float64) where A
        return new{A}(data, preconditioner, prev, df)
    end
    function OptimState(data::A) where A
        return new{A}(data, nothing, NaN, NaN)
    end
end

function ground_state_new(H::MultiBosonLiebLiniger, ψ0::MultiBosonCMPSData_MDMinv; do_preconditioning::Bool=true, maxiter::Int=10000, gradtol=1e-6, fϵ=(x->1e-3*x))
    if H.L < Inf
        error("finite size not implemented yet.")
    end

    function fE_inf(x::OptimState{MultiBosonCMPSData_MDMinv})
        ψ = x.data
        ψn = CMPSData(ψ)
        OH = kinetic(ψn) + H.cs[1,1]* point_interaction(ψn, 1) + H.cs[2,2]* point_interaction(ψn, 2) + H.cs[1,2] * point_interaction(ψn, 1, 2) + H.cs[2,1] * point_interaction(ψn, 2, 1) - H.μs[1] * particle_density(ψn, 1) - H.μs[2] * particle_density(ψn, 2)
        TM = TransferMatrix(ψn, ψn)
        envL = permute(left_env(TM), (), (1, 2))
        envR = permute(right_env(TM), (2, 1), ()) 
        return real(tr(envL * OH * envR) / tr(envL * envR))
    end
    
    function fgE(x::OptimState{MultiBosonCMPSData_MDMinv})
        ψ = x.data
        E, ∂ψ = withgradient(fE_inf, ψ)
        g = diff_to_grad(ψ, ∂ψ[1])
        return E, g
    end
    
    function inner(x, a::MultiBosonCMPSData_MDMinv_Grad, b::MultiBosonCMPSData_MDMinv_Grad)
        # be careful the cases with or without a factor of 2. depends on how to define the complex gradient
        return real(dot(a, b)) 
    end

    function retract(x::OptimState{MultiBosonCMPSData_MDMinv}, dψ::MultiBosonCMPSData_MDMinv_Grad, α::Real)
        ψ1 = retract_left_canonical(ψ, α, dψ.dDs, dψ.X)
        return OptimState(ψ1, x.preconditioner, x.prev, x.df), dψ
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
    function _no_precondition(x::OptimState{MultiBosonCMPSData_MDMinv}, dψ::MultiBosonCMPSData_MDMinv_Grad)
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
    function _precondition(x0::OptimState{MultiBosonCMPSData_MDMinv}, dψ::MultiBosonCMPSData_MDMinv_Grad)
        ϵ = isnan(x0.df) ? max(1e-12, fϵ(norm(dψ))) : max(1e-12, fϵ(x0.df))
        χ, d = get_χ(ψ0), get_d(ψ0)

        if isnothing(x0.preconditioner)
            P = zeros(ComplexF64, χ^2+d*χ, χ^2+d*χ)

            blas_num_threads = LinearAlgebra.BLAS.get_num_threads()
            LinearAlgebra.BLAS.set_num_threads(1)
            Threads.@threads for ix in 1:(χ^2+d*χ)
                v = zeros(ComplexF64, χ^2+d*χ)
                v[ix] = 1
                g = MultiBosonCMPSData_MDMinv_Grad(v, χ, d)
                g1 = tangent_map(ψ0, g)
                P[:, ix] = vec(g1)
            end 
            LinearAlgebra.BLAS.set_num_threads(blas_num_threads)
            P[diagind(P)] .+= ϵ
            x0.preconditioner = P
        end
        vp = x0.preconditioner \ vec(dψ)
        return MultiBosonCMPSData_MDMinv_Grad(vp, χ, d)
    end

    transport!(v, x, d, α, xnew) = v

    function finalize!(x::OptimState{MultiBosonCMPSData_MDMinv}, f, g, numiter)
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
    ψ1, E1, grad1, numfg1, history1 = optimize(fgE, ψ0, optalg_LBFGS; retract = retract,
                                    precondition = precondition,
                                    inner = inner, transport! =transport!,
                                    scale! = scale!, add! = add!, finalize! = finalize!
                                    );

    res = (ψ1.data, E1, grad1, numfg1, history1)
    return res
end
