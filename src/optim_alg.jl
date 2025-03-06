mutable struct OptimState{A}
    data::A
    preconditioner::Any
    prev::Float64
    df::Float64

    function OptimState(data::A, preconditioner, prev::Float64, df::Float64) where A
        return new{A}(data, preconditioner, prev, df)
    end
    function OptimState(data::A) where A
        return new{A}(data, missing, NaN, NaN)
    end
end

abstract type Algorithm end 

struct CircularCMPSRiemannian <: Algorithm
    maxiter::Int
    tol::Real 
    verbosity::Int
end

function minimize(_f, init::CMPSData, alg::CircularCMPSRiemannian; finalize! = OptimKit._finalize!, fϵ=identity)
    
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
    function precondition(x::OptimState{CMPSData}, dϕ::CMPSData)
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

            x.preconditioner = herm_reg_inv(vr, ϵ) 
        end

        Q = dϕ.Q  
        Rs = dϕ.Rs .* Ref(x.preconditioner)

        return CMPSData(Q, Rs)
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
            @inline function kron1(A::TensorMap, B::TensorMap)
                @tensor AB[-1 -2; -3 -4] := A[-3; -1] * B[-2; -4]
                return AB
            end
            P1 = kron1(Id, ϕ.Rs[2] * vr * ϕ.Rs[2]') + 
                 kron1(ϕ.Rs[2]' * ϕ.Rs[2], vr) + 
                 kron1(ϕ.Rs[2], vr * ϕ.Rs[2]') + 
                 kron1(ϕ.Rs[2]', ϕ.Rs[2] * vr)
            P2 = kron1(Id, ϕ.Rs[1] * vr * ϕ.Rs[1]') + 
                 kron1(ϕ.Rs[1]' * ϕ.Rs[1], vr) + 
                 kron1(ϕ.Rs[1], vr * ϕ.Rs[1]') + 
                 kron1(ϕ.Rs[1]', ϕ.Rs[1] * vr)

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

    x, fvalue, grad, numfg, history = optimize(_fg, OptimState(init), optalg_LBFGS; retract = retract, precondition = precondition, inner = inner, transport! = transport!, scale! = scale!, add! = add!, finalize! = finalize_wrapped!)

    return x.data, fvalue, grad, numfg, history
end

struct OptimNumber <: Algorithm
    maxiter::Int
    tol::Real 
    verbosity::Int
end

function minimize(_f, init::Number, alg::OptimNumber)
    
    function _fg(x::Number)
        return _f(x), _f'(x) 
    end
    function inner(x, x1::Number, x2::Number)
        return real(x1' * x2)
    end
    function retract(x::Number, dx::Number, α::Real)
        return x+ α*dx, dx
    end
    function scale!(dx::Number, α::Number)
        dx *= α
        return dx
    end
    function add!(dx::Number, dx1::Number, α::Number)
        dx += dx1 * α
        return dx
    end
    transport!(v, x, d, α, xnew) = v
    
    optalg_LBFGS = LBFGS(;maxiter=alg.maxiter, gradtol=alg.tol, verbosity=alg.verbosity)

    ψopt, fvalue, grad, numfg, history = optimize(_fg, init, optalg_LBFGS; retract=retract,  inner=inner, transport! =transport!, scale! =scale!, add! =add!)

    return ψopt, fvalue, grad, numfg, history
end