function energy(H::MultiBosonLiebLiniger, ψ::CMPSData, Λ::Real; order::Integer=1)
    OH = kinetic(ψ) + H.cs[1,1]* point_interaction(ψ, 1) + H.cs[2,2]* point_interaction(ψ, 2) + H.cs[1,2] * point_interaction(ψ, 1, 2) + H.cs[2,1] * point_interaction(ψ, 2, 1) - H.μs[1] * particle_density(ψ, 1) - H.μs[2] * particle_density(ψ, 2) + penalty_term(ψ, 1, 2, Λ; order=order)
    TM = TransferMatrix(ψ, ψ)
    envL = permute(left_env(TM), (), (1, 2))
    envR = permute(right_env(TM), (2, 1), ()) 
    return real(tr(envL * OH * envR) / tr(envL * envR))
end
# FIXME. AD for this function is broken. seems to be due to that the ed rule for exp is not correct.
function energy2(H::MultiBosonLiebLiniger, ψ::CMPSData, Λ::Real)
    OH = kinetic(ψ) + H.cs[1,1]* point_interaction(ψ, 1) + H.cs[2,2]* point_interaction(ψ, 2) + H.cs[1,2] * point_interaction(ψ, 1, 2) + H.cs[2,1] * point_interaction(ψ, 2, 1) - H.μs[1] * particle_density(ψ, 1) - H.μs[2] * particle_density(ψ, 2) + penalty_term_type2(ψ, 1, 2, Λ)
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

function minimize1(_f, init::CMPSData, alg::CircularCMPSRiemannian; finalize! = OptimKit._finalize!, fϵ=identity, Λ::Real=1.0)
    
    function _fg(x::OptimState{CMPSData})
        ϕ = x.data
        fvalue = _f(ϕ)
        ∂ϕ = _f'(ϕ)
        dQ = zero(∂ϕ.Q) 
        dRs = ∂ϕ.Rs .- ϕ.Rs .* Ref(∂ϕ.Q)

        return fvalue, CMPSData(dQ, dRs) 
    end
    function inner(x, ϕ1::CMPSData, ϕ2::CMPSData)
        return real(sum(dot.(ϕ1.Rs, ϕ2.Rs)))
    end
    function retract(x::OptimState{CMPSData}, dϕ::CMPSData, α::Real)
        ϕ = x.data
        Rs = ϕ.Rs .+ α .* dϕ.Rs 
        Q = ϕ.Q - α * sum(adjoint.(ϕ.Rs) .* dϕ.Rs) - 0.5 * α^2 * sum(adjoint.(dϕ.Rs) .* dϕ.Rs)
        ϕ1 = CMPSData(Q, Rs)

        return OptimState(ϕ1, missing, x.prev, x.df), dϕ
    end
    function scale!(dϕ::CMPSData, α::Number)
        dϕ.Q = dϕ.Q * α
        dϕ.Rs .= dϕ.Rs .* α
        return dϕ
    end
    function add!(dϕ::CMPSData, dϕ1::CMPSData, α::Number)
        dϕ.Q += dϕ1.Q * α
        dϕ.Rs .+= dϕ1.Rs .* α
        return dϕ
    end
    function precondition1(x::OptimState{CMPSData}, dϕ::CMPSData)
        ϕ = x.data

        if ismissing(x.preconditioner)
            δ = inner(ϕ, dϕ, dϕ)
            ϵ = isnan(x.df) ? 1e-3*fϵ(sqrt(δ)) : fϵ(x.df)
            ϵ = max(1e-12, ϵ)

            fK = transfer_matrix(ϕ, ϕ)

            # solve the fixed point equation
            init = similar(ϕ.Q, _firstspace(ϕ.Q)←_firstspace(ϕ.Q))
            randomize!(init);
            _, vrs, _ = eigsolve(fK, init, 1, :LR)
            vr = vrs[1]

            Id = id(_firstspace(ϕ.Q))
            @inline function kron1(A::AbstractTensorMap, B::AbstractTensorMap)
                @tensor AB[-1 -2; -3 -4] := A[-3; -1] * B[-2; -4]
                return AB
            end

            P1 = kron1(Id, ϕ.Rs[2] * vr * ϕ.Rs[2]') + 
                 kron1(ϕ.Rs[2]' * ϕ.Rs[2], vr) - 
                 kron1(ϕ.Rs[2], vr * ϕ.Rs[2]') - 
                 kron1(ϕ.Rs[2]', ϕ.Rs[2] * vr) + 
                 (1/Λ) * kron1(Id, vr)

            P2 = kron1(Id, ϕ.Rs[1] * vr * ϕ.Rs[1]') + 
                 kron1(ϕ.Rs[1]' * ϕ.Rs[1], vr) - 
                 kron1(ϕ.Rs[1], vr * ϕ.Rs[1]') - 
                 kron1(ϕ.Rs[1]', ϕ.Rs[1] * vr) + 
                 (1/Λ) * kron1(Id, vr)

            x.preconditioner = [herm_reg_inv(P1, ϵ), herm_reg_inv(P2, ϵ)]
        end

        Q = dϕ.Q  
        @tensor R1[-1; -2] := x.preconditioner[1][1 2; -1 -2] * dϕ.Rs[1][1; 2]
        @tensor R2[-1; -2] := x.preconditioner[2][1 2; -1 -2] * dϕ.Rs[2][1; 2]

        return CMPSData(Q, [R1, R2])
    end

    transport!(v, x, d, α, xnew) = v

    function finalize_wrapped!(x::OptimState{CMPSData}, f, g, numiter)
        x.preconditioner = missing
        x.df = abs(f - x.prev)
        x.prev = f
        return finalize!(x, f, g, numiter)
    end
    
    optalg_LBFGS = LBFGS(;maxiter=alg.maxiter, gradtol=alg.tol, verbosity=alg.verbosity)

    init = left_canonical(init)[2] # ensure the input is left canonical

    x, fvalue, grad, numfg, history = optimize(_fg, OptimState(init), optalg_LBFGS; retract = retract, precondition = precondition1, inner = inner, transport! = transport!, scale! = scale!, add! = add!, finalize! = finalize_wrapped!)

    return x.data, fvalue, grad, numfg, history
end

"""
    ground_state(H::MultiBosonLiebLiniger, ψ0::CMPSData; Λ::Real=1.0)

Find the ground state of the MultiBosonLiebLiniger model with the given Hamiltonian `H` and the initial state `ψ0`. 
The multi-boson cMPS is parametrized with no regularity conditions. This allows a more efficient optimization. 
The regularity condition is achieved via an additional Lagrangian multiplier term `Λ [ψ1, ψ2]† [ψ1, ψ2]`.
"""
function ground_state(H::AbstractHamiltonian, ψ0::CMPSData; Λs::Vector{<:Real}=10 .^ (2:(1/3):5), gradtol=1e-2, maxiter=100, do_prerun=true, fϵ=(x->10*x), order::Integer=1)
    if H.L < Inf
        error("finite size not implemented yet.")
    end

    function fE_inf(ψ::CMPSData, Λ::Real; order::Integer=1) 
        return energy(H, ψ, Λ; order=order)
    end
    #function fE_inf(ψ::CMPSData, Λ::Real) 
    #    return energy2(H, ψ, Λ)
    #end
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

        #if numiter > 10 && err > err_history[end-1] 
        #    println("err start to increase. stop the optimization.")
        #    return x, f, 0*g, numiter
        #end

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

    ψ0 = deepcopy(ψ)
    #if do_prerun
    #    Λi = Λs[1] ^ 2 / Λs[2] 
    #    ψ, E, grad, numfg, history = minimize1(x->fE(x, Λi), ψ, CircularCMPSRiemannian(maxiter, 1e-2, 2); finalize! = OptimKit._finalize!, fϵ=fϵ, Λ=Λi)
    #    ψ0 = deepcopy(ψ)
    #end
    E_history, gnorm_history, err_history = Float64[], Float64[], Float64[]

    order1 = 1
    for Λ in Λs
        println("Doing simulation for lg Λ = $(log10(Λ)), penalty order = $order1")
        ψ, E, grad, numfg, history = minimize1(x->fE(x, Λ), ψ, CircularCMPSRiemannian(maxiter, gradtol, 1); finalize! = _finalize!, fϵ=fϵ, Λ=Λ)
        if Λ ≈ Λs[1]
            ψ0 = deepcopy(ψ)
        end
        total_numfg += numfg
       
        #while 1 < Λ * norm(ψ.Rs[1] * ψ.Rs[2] - ψ.Rs[2] * ψ.Rs[1])^2 < (1e3)^(1/order1)
        #    println("Doing simulation for lg Λ = $(log10(Λ)), penalty order = $order1")
        #    order1 += 1
        #    ψ, E, grad, numfg, history = minimize(x->fE(x, Λ; order=order1), ψ, CircularCMPSRiemannian(maxiter, gradtol, 1); finalize! = _finalize!, fϵ=fϵ)
        #    total_numfg += numfg
        #end
    end
    #return optimal_solution, optimal_E, optimal_grad, total_numfg, hcat(E_history, gnorm_history, err_history)
    return ψ, ψ0, E, grad, total_numfg, hcat(E_history, gnorm_history, err_history)
end