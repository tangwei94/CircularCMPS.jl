using LinearAlgebra, TensorKit, KrylovKit
using Revise
using CircularCMPS 

function inner1(ψ::CMPSData, ϕ::CMPSData)
    return real(sum(dot.(ψ.Rs, ϕ.Rs))) + real(dot(ψ.Q, ϕ.Q))
end
_firstspace = CircularCMPS._firstspace

ψ = CMPSData(rand, 3, 2);

L = 12.321

ψrand = CMPSData(rand, 2, 2)
result1 = compress(ψ, 2, L; init=ψrand)
results = compress(ψ, 2, L)

expK, ln_norm = finite_env(ψ*ψ, L)

    # variational optimization
    function _f(ϕ::CMPSData)
        return -real(ln_ovlp(ϕ, ψ, L) + ln_ovlp(ψ, ϕ, L) - ln_ovlp(ϕ, ϕ, L) - ln_norm)
    end
    function _fg(ϕ::CMPSData)
        fvalue = _f(ϕ)
        ∂ϕ = _f'(ϕ)
        dQ = zero(∂ϕ.Q) 
        dRs = ∂ϕ.Rs .- ϕ.Rs .* Ref(∂ϕ.Q)

        return fvalue, CMPSData(dQ, dRs) 
    end
    function inner(ϕ, ϕ1::CMPSData, ϕ2::CMPSData)
        return real(sum(dot.(ϕ1.Rs, ϕ2.Rs)))
    end
    function retract(ϕ::CMPSData, dϕ::CMPSData, α::Real)
        Rs = ϕ.Rs .+ α .* dϕ.Rs 
        Q = ϕ.Q - α * sum(adjoint.(ϕ.Rs) .* dϕ.Rs) - 0.5 * α^2 * sum(dϕ.Rs' .* dϕ.Rs)
        ϕ1 = CMPSData(Q, Rs)
        return ϕ1, dϕ
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
    function precondition(ϕ::CMPSData, dϕ::CMPSData)
        fK = transfer_matrix(ϕ, ϕ)

        # solve the fixed point equation
        init = similar(ϕ.Q, _firstspace(ϕ.Q)←_firstspace(ϕ.Q))
        randomize!(init);
        _, vrs, _ = eigsolve(fK, init, 1, :LR)
        vr = vrs[1]

        δ = inner(ϕ, dϕ, dϕ)

        P = herm_reg_inv(vr, max(1e-12, 1e-3*δ)) 

        Q = dϕ.Q
        Rs = dϕ.Rs .* Ref(P)

        return CMPSData(Q, Rs)
    end
    transport!(v, x, d, α, xnew) = v

ψ1 = left_canonical(ψ1)[2]
_f(ψ1)

fidel, gϕ = _fg(ψ1)
dϕ = precondition(ψ1, gϕ)

α = 0
Δα = 1e-6
ϕ0, _ = retract(ψ1, dϕ, α)
ϕ1, _ = retract(ψ1, dϕ, α-Δα)
ϕ2, _ = retract(ψ1, dϕ, α+Δα)

fidel0, gϕ0 = _fg(ϕ0)
fidel1, gϕ1 = _fg(ϕ1)
fidel2, gϕ2 = _fg(ϕ2)

(fidel2 - fidel1) / (2*Δα)
inner(ϕ0, gϕ0, dϕ)

V = - sum(adjoint.(ϕ0.Rs) .* dϕ.Rs)
Ws = dϕ.Rs
dϕ11 = CMPSData(V, Ws)
_fa(a) = _f(ϕ0 + a * dϕ11)
inner1(_f'(ϕ0), dϕ11)

for δ in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    g = (_fa(δ) - _fa(-δ)) / (2*δ)
    @show norm(g- inner1(_f'(ϕ0), dϕ11))

    g2 = (_f(retract(ϕ0, dϕ, δ)[1]) - _f(retract(ϕ0, dϕ, -δ)[1])) / (2*δ)
    @show norm(g2- inner1(_f'(ϕ0), dϕ11) )
end

inner(0, gϕ0, dϕ)
