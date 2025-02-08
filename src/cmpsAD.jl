#"""
#    rrule(::typeof(tr), A::AbstractTensorMap)
#
#    rrule for `TensorKit.tr`.
#"""
#function ChainRulesCore.rrule(::typeof(tr), A::AbstractTensorMap)
#    fwd = tr(A)
#    function tr_pushback(f̄wd)
#        Ā = f̄wd * id(domain(A))
#        return NoTangent(), Ā
#    end 
#    return fwd, tr_pushback
#end

#"""
#    rrule(::typeof(adjoint), A::AbstractTensorMap)
#
#    rrule for adjoint.
#"""
#function ChainRulesCore.rrule(::typeof(adjoint), A::AbstractTensorMap)
#    function adjoint_pushback(f̄wd)
#        return NoTangent(), f̄wd'
#    end
#    return A', adjoint_pushback
#end

@non_differentiable id(V::VectorSpace)
@non_differentiable isomorphism(cod::VectorSpace, dom::VectorSpace)
@non_differentiable domain(W::TensorKit.HomSpace)
@non_differentiable domain(t::AbstractTensorMap)
@non_differentiable domain(t::AbstractTensorMap, i)
@non_differentiable codomain(W::TensorKit.HomSpace)
@non_differentiable codomain(t::AbstractTensorMap)
@non_differentiable codomain(t::AbstractTensorMap, i)

"""
    ChainRulesCore.rrule(::typeof(getfield), ψ::CMPSData, S::Symbol)

    rrule for getfield.
"""
function ChainRulesCore.rrule(::typeof(getfield), ψ::CMPSData, S::Symbol)
    # TODO. doesn't work. zygote doesn't call this function. to fix.
    fwd = getfield(ψ, S)
    Q0 = zero(ψ.Q)
    R0s = [zero(R) for R in ψ.Rs]
    function getfield_pushback(f̄wd)
        if S == :Q 
            return NoTangent(), CMPSData(f̄wd, R0s), NoTangent()
        elseif S == :Rs 
            return NoTangent(), CMPSData(Q0, f̄wd), NoTangent()
        else
            return NoTangent(), NoTangent(), NoTangent()
        end
    end 
    return fwd, getfield_pushback
end

"""
    ChainRulesCore.rrule(::typeof(get_matrices), ψ::CMPSData)

    rrule for get_matrices.
"""
function ChainRulesCore.rrule(::typeof(get_matrices), ψ::CMPSData)
    fwd = get_matrices(ψ)
    d = get_d(ψ)
    Q0 = zero(ψ.Q)
    R0s = [zero(R) for R in ψ.Rs]
    function get_matrices_pushback(f̄wd)
        (Q̄, R̄s) = f̄wd
        (Q̄ isa ZeroTangent) && (Q̄ = Q0)
        for ix in 1:d
            (R̄s[ix] isa ZeroTangent) && (R̄s[ix] = R0s[ix])
        end
        return NoTangent(), CMPSData(Q̄, typeof(R0s)(R̄s))
    end
    return fwd, get_matrices_pushback
end

"""
    rrule(::Type{CMPSData}, Q::MPSBondTensor, Rs::Vector{MPSBondTensor})

    rrule for the constructor of `cmps`.
"""
function ChainRulesCore.rrule(::Type{CMPSData}, Q::MPSBondTensor, Rs::Vector{<:MPSBondTensor})
    function cmps_pushback(f̄wd)
        return NoTangent(), f̄wd.Q, f̄wd.Rs
    end
    return CMPSData(Q, Rs), cmps_pushback
end

"""
    rrule(::typeof(K_mat), phi::cmps, psi::cmps) 

    The reverse rule for function K_mat.  
"""
function ChainRulesCore.rrule(::typeof(K_mat), ϕ::CMPSData, ψ::CMPSData)  
    Id_ψ, Id_ϕ = id(space(ψ)), id(space(ϕ)) 
    fwd = K_mat(ϕ, ψ) 

    function K_mat_pushback(f̄wd)
        f̄wd = K_permute(f̄wd)
        @tensor Q̄_ϕ[-1; -2] := conj(f̄wd[1 -2; 1 -1])
        @tensor Q̄_ψ[-1; -2] := f̄wd[-1 1; -2 1]
        R̄s_ψ, R̄s_ϕ = MPSBondTensor[], MPSBondTensor[]
        for (Rψ, Rϕ) in zip(ψ.Rs, ϕ.Rs)
            @tensor R̄ϕ[-1; -2] := conj(f̄wd[2 -2; 1 -1]) * conj(Rψ'[1; 2]) # TODO specifically test this one
            @tensor R̄ψ[-1; -2] := f̄wd[-1 2; -2 1] * Rϕ[1; 2]
            push!(R̄s_ψ, R̄ψ)
            push!(R̄s_ϕ, R̄ϕ)
        end
        ϕ̄ = CMPSData(Q̄_ϕ, R̄s_ϕ)
        ψ̄ = CMPSData(Q̄_ψ, R̄s_ψ)
        return NoTangent(), ϕ̄, ψ̄
    end
    return fwd, K_mat_pushback
end

"""
    rrule(::typeof(finite_env), K::AbstractTensorMap{T, S, 2, 2}, L::Real)

    Backward rule for `finite_env`.
    See https://math.stackexchange.com/a/3868894/488003 and https://doi.org/10.1006/aama.1995.1017 for the gradient of exp(t)
"""
function ChainRulesCore.rrule(::typeof(finite_env), K::AbstractTensorMap{T, S, 2, 2}, L::Real) where {T, S}
    W, UR = eig(K)
    UL = inv(UR)
    Ws = []

    if W.data isa Matrix 
        Ws = diag(W.data)
    elseif W.data isa TensorKit.SortedVectorDict
        Ws = vcat([diag(values) for (_, values) in W.data]...)
    end
    Wmax = maximum(real.(Ws))
    ln_of_norm = Wmax * L + log(sum(exp.(Ws .* L .- Wmax * L)))

    W = W - (ln_of_norm / L) * id(_firstspace(W)) 
    expK = UR * exp(W * L) * UL
    
    Ws = Ws .- ln_of_norm / L

    function finite_env_pushback(f̄wd)
        ēxpK, l̄n_norm = f̄wd 
       
        K̄ = zero(K)

        if ēxpK != ZeroTangent()
            if W.data isa TensorKit.SortedVectorDict
                # TODO. symmetric tensor
                error("symmetric tensor. not implemented")
            end
            function coeff(a::Number, b::Number) 
                if a ≈ b
                    return L*exp(a*L)
                else 
                    return (exp(a*L) - exp(b*L)) / (a - b)
                end
            end
            M = UR' * ēxpK * UL'
            M1 = similar(M)
            copyto!(M1.data, M.data .* coeff.(Ws', conj.(Ws)))
            K̄ += UL' * M1 * UR' - L * tr(ēxpK * expK') * expK'
        end
        if l̄n_norm != ZeroTangent()
            K̄ += l̄n_norm * L * expK'
        end
        return NoTangent(), K̄, NoTangent()
    end 
    return (expK, ln_of_norm), finite_env_pushback
end
#function ChainRulesCore.rrule(::typeof(finite_env), K::AbstractTensorMap{T, S, 2, 2}, L::Real) where {T, S}
#    # TODO. gaussian quadrature. not efficient. parallelize?
#    expK, ln_norm = finite_env(K, L)
#
#    function finite_env_pushback(f̄wd)
#        ēxpK, l̄n_norm = f̄wd
#        
#        xs, ws = gausslegendre(80)
#        xs = (xs .+ 1) .* (L/2)
#        ws = ws .* (L/2)
#
#        IK = id(domain(K))
#        Knm = K - (ln_norm / L) * IK
#
#        K̄ = zero(K)
#        for (xᵢ, wᵢ) in zip(xs, ws)
#            K1 = exponentiate(x -> Knm' * x, xᵢ, ēxpK)[1]
#            K2 = exponentiate(x -> x * Knm', L - xᵢ, K1)[1]
#            K̄ = K̄ + wᵢ * K2
#        end
#        K̄ = K̄ - L * tr(ēxpK * expK') * expK'
#
#        return NoTangent(), K̄, NoTangent()
#    end
#
#    return (expK, ln_norm), finite_env_pushback
#end