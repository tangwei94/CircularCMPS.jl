function energy(H::MultiBosonLiebLiniger, ψ::CMPSData, Λ::Real; order::Integer=1)
    OH = kinetic(ψ) + H.cs[1,1]* point_interaction(ψ, 1) + H.cs[2,2]* point_interaction(ψ, 2) + H.cs[1,2] * point_interaction(ψ, 1, 2) + H.cs[2,1] * point_interaction(ψ, 2, 1) - H.μs[1] * particle_density(ψ, 1) - H.μs[2] * particle_density(ψ, 2) + penalty_term(ψ, 1, 2, Λ; order=order)
    TM = TransferMatrix(ψ, ψ)
    envL = permute(left_env(TM), (), (1, 2))
    envR = permute(right_env(TM), (2, 1), ()) 
    return real(tr(envL * OH * envR) / tr(envL * envR))
end

function energy(H::MultiBosonLiebLinigerWithPairing, ψ::CMPSData, Λ::Real; order::Integer=1)
    OH = kinetic(ψ) + H.cs[1,1]* point_interaction(ψ, 1) + H.cs[2,2]* point_interaction(ψ, 2) + H.cs[1,2] * point_interaction(ψ, 1, 2) + H.cs[2,1] * point_interaction(ψ, 2, 1) - H.μs[1] * particle_density(ψ, 1) - H.μs[2] * particle_density(ψ, 2) + H.us[1] * pairing(ψ, 1) + H.us[2] * pairing(ψ, 2) + penalty_term(ψ, 1, 2, Λ; order=order)
    TM = TransferMatrix(ψ, ψ)
    envL = permute(left_env(TM), (), (1, 2))
    envR = permute(right_env(TM), (2, 1), ()) 
    return real(tr(envL * OH * envR) / tr(envL * envR))
end

"""
    ground_state(H::MultiBosonLiebLiniger, ψ0::CMPSData; Λ::Real=1.0)

Find the ground state of the MultiBosonLiebLiniger model with the given Hamiltonian `H` and the initial state `ψ0`. 
The multi-boson cMPS is parametrized with no regularity conditions. This allows a more efficient optimization. 
The regularity condition is achieved via an additional Lagrangian multiplier term `Λ [ψ1, ψ2]† [ψ1, ψ2]`.
"""
function ground_state(H::AbstractHamiltonian, ψ0::CMPSData; Λs::Vector{<:Real}=10 .^ (2:(1/3):5), gradtol=1e-2, maxiter=100, do_prerun=true, fϵ=(x->10*x), energy_prefactor=1.0, order::Integer=1)
    if H.L < Inf
        error("finite size not implemented yet.")
    end

    function fE_inf(ψ::CMPSData, Λ::Real) 
        return energy(H, ψ, Λ; order=order)
    end
    #function fE_finiteL(ψ::CMPSData, Λ::Real)
    #    OH = kinetic(ψ) + H.cs[1,1]* point_interaction(ψ, 1) + H.cs[2,2]* point_interaction(ψ, 2) + H.cs[1,2] * point_interaction(ψ, 1, 2) + H.cs[2,1] * point_interaction(ψ, 2, 1) - H.μs[1] * particle_density(ψ, 1) - H.μs[2] * particle_density(ψ, 2) + lagrangian_multiplier(ψ, 1, 2, Λ) 
    #    expK, _ = finite_env(K_mat(ψ, ψ), H.L)
    #    return real(tr(expK * OH))
    #end 
    function _finalize!(x, f, g, numiter)
        #x1, err = convert_to_MultiBosonCMPSData_MDMinv(x.data)
        #x1 = left_canonical(x1)
        #fMDMinv = (xm -> fE(CMPSData(xm), 0))
        #f1, diff1 = withgradient(fMDMinv, x1)
        #g1 = diff_to_grad(x1, diff1[1])
        push!(E_history, f)
        push!(gnorm_history, norm(g))
        # only works for two types of bosons
        R1, R2 = x.data.Rs[1], x.data.Rs[2]
        err = norm(R1 * R2 - R2 * R1)
        push!(err_history, err)
        println("$numiter: E: $f gnorm: $(norm(g)) err: $err ratio $(x.df/norm(g))")

        if numiter > 10 && err > err_history[end-1] && (energy_prefactor > 1e-12)
            println("err start to increase. stop the optimization.")
            return x, f, 0*g, numiter
        end

        # NOTE. choosing the optimal solution according to energy is suboptimal since the lagrange multiplier process sometimes increases the energy, while we tend to trust the result after the lagrange multiplier process.
        #if f1 < optimal_E
        #    optimal_solution = deepcopy(x1)
        #    optimal_E = f1
        #    optimal_grad = deepcopy(g1)
        #end

        return x, f, g, numiter
    end

    fE = (H.L == Inf) ? fE_inf : fE_finiteL

    ψ = left_canonical(ψ0)[2]
    E, grad = withgradient(x->fE_inf(x, Λs[1]), ψ)
    grad = grad[1]
    total_numfg = 0

    if do_prerun
        Λi = Λs[1] ^ 2 / Λs[2] 
        ψ, E, grad, numfg, history = minimize(x->fE(x, Λi), ψ, CircularCMPSRiemannian(maxiter, gradtol, 2); finalize! = OptimKit._finalize!, fϵ=fϵ)
    end
    E_history, gnorm_history, err_history = Float64[], Float64[], Float64[]

    for Λ in Λs
        println("Doing simulation for lg Λ = $(log10(Λ))")
        ψ, E, grad, numfg, history = minimize(x->fE(x, Λ), ψ, CircularCMPSRiemannian(maxiter, gradtol, 1); finalize! = _finalize!, fϵ=fϵ)
        total_numfg += numfg
    end
    #return optimal_solution, optimal_E, optimal_grad, total_numfg, hcat(E_history, gnorm_history, err_history)
    return ψ, E, grad, total_numfg, hcat(E_history, gnorm_history, err_history)
end