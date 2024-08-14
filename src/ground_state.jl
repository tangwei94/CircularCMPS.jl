# TODO. construct a type for Hamiltonians

function lieb_liniger_ground_state(c::Real, μ::Real, L::Real, ψ0::Union{CMPSData, Nothing}=nothing)
    function fE(ψ::CMPSData)
        OH = kinetic(ψ) + c*point_interaction(ψ) - μ * particle_density(ψ)
        expK, _ = finite_env(K_mat(ψ, ψ), L)
        return real(tr(expK * OH))
    end

    function fgE(ψ::CMPSData)
        E = fE(ψ)
        ∂ψ = fE'(ψ) 
        dQ = zero(∂ψ.Q) #- sum(ψ.Rs' .* ∂ψ.Rs)
        dRs = ∂ψ.Rs .- ψ.Rs .* Ref(∂ψ.Q) #the second term makes sure it is a true gradient! 

        return E, CMPSData(dQ, dRs)
    end

    function inner(ψ, ψ1::CMPSData, ψ2::CMPSData)
        # be careful the cases with or without a factor of 2. depends on how to define the complex gradient
        return real(sum(dot.(ψ1.Rs, ψ2.Rs))) 
    end

    function retract(ψ::CMPSData, dψ::CMPSData, α::Real)
        Rs = ψ.Rs .+ α .* dψ.Rs 
        Q = ψ.Q - α * sum(ψ.Rs' .* dψ.Rs) - 0.5 * α^2 * sum(dψ.Rs' .* dψ.Rs)
        ψ1 = CMPSData(Q, Rs)
        #ψ1 = left_canonical(ψ1)[2]
        return ψ1, dψ
    end

    function scale!(dψ::CMPSData, α::Number)
        dψ.Q = dψ.Q * α
        dψ.Rs .= dψ.Rs .* α
        return dψ
    end

    function add!(dψ::CMPSData, dψ1::CMPSData, α::Number) 
        dψ.Q += dψ1.Q * α
        dψ.Rs .+= dψ1.Rs .* α
        return dψ
    end

    # only for comparison
    #function no_precondition(ψ::CMPSData, dψ::CMPSData)
    #    return dψ
    #end

    function precondition(ψ::CMPSData, dψ::CMPSData)
        fK = transfer_matrix(ψ, ψ)

        # solve the fixed point equation
        init = similar(ψ.Q, _firstspace(ψ.Q)←_firstspace(ψ.Q))
        randomize!(init);
        _, vrs, _ = eigsolve(fK, init, 1, :LR)
        vr = vrs[1]

        δ = inner(ψ, dψ, dψ)

        P = herm_reg_inv(vr, max(1e-12, 1e-3*δ)) 

        Q = dψ.Q  
        Rs = dψ.Rs .* Ref(P)

        return CMPSData(Q, Rs)
    end

    transport!(v, x, d, α, xnew) = v

    optalg_LBFGS = LBFGS(;maxiter=10000, gradtol=1e-6, verbosity=2)

    ψ = left_canonical(ψ0)[2]
    ψ1, E, grad, numfg, history = optimize(fgE, ψ, optalg_LBFGS; retract = retract,
                                    precondition = precondition,
                                    inner = inner, transport! =transport!,
                                    scale! = scale!, add! = add!
                                    );
    return ψ1, E, grad, numfg, history 

end

