# copied from MPSKit.jl
const MPSBondTensor{T,S} = AbstractTensorMap{T,S,1,1} 
const GenericMPSTensor{T,S,N} = AbstractTensorMap{T,S,N,1}  
const MPSTensor{T,S} = GenericMPSTensor{T,S,2} 

_firstspace(t::AbstractTensorMap) = space(t, 1)
_lastspace(t::AbstractTensorMap) = space(t, numind(t))

#=
map every element in the tensormap to dfun(E)
allows us to create random tensormaps for any storagetype
=#
function fill_data!(a::TensorMap,dfun)
    E = eltype(a);

    for (k,v) in blocks(a)
        map!(x->dfun(E),v,v);
    end

    a
end
randomize!(a::TensorMap) = fill_data!(a,randn)

# default permutation of K matrix 
function K_permute(K::AbstractTensorMap{T, S, 2, 2}) where {T,S}
    return permute(K, (2, 3), (4, 1))
end

function K_permute_back(K::AbstractTensorMap{T, S, 2, 2}) where {T,S}
    return permute(K, (4, 1), (2, 3))
end

function K_otimes(A::MPSBondTensor, B::MPSBondTensor)
    @tensor Abar_otimes_B[-1, -2; -3, -4] := A'[-3, -1] * B[-2, -4]
    return Abar_otimes_B
end

function Kact_R(K::AbstractTensorMap{T, S, 2, 2}, v::MPSBondTensor) where {T, S}
    @tensor Kr[-1; -2] := K[-1 2; 1 -2] * v[1; 2]
    return Kr
end

function Kact_L(K::AbstractTensorMap{T, S, 2, 2}, v::MPSBondTensor) where {T, S}
    @tensor Kl[-1; -2] := K[1 -1; -2 2] * v[2; 1]
    return Kl
end

function herm_reg_inv(A::AbstractTensorMap, δ::Real)
    # A is Hermitian up to a phase

    _, S, V = tsvd(A)
    Id = id(_firstspace(S))
    Ainv = V' * (inv(S + δ*Id)) * V

    return Ainv

end

function quasi_inv(A::AbstractTensorMap, δ::Real)
    U, S, V = tsvd(A)
    function f_pseudo_inv(x::Number)
        if norm(x) < δ
            return x
        else
            return 1/x
        end
    end
    
    for (k,v) in blocks(S)
        map!(x->f_pseudo_inv(x), v, v);
    end
    return V' * S * U'
end

function ChainRulesCore.rrule(::typeof(TensorKit.exp), K::TensorMap)
    W, UR = eig(K)
    UL = inv(UR)
    Ws = []

    if W.data isa Matrix 
        Ws = diag(W.data)
    elseif W.data isa TensorKit.SortedVectorDict
        Ws = vcat([diag(values) for (_, values) in W.data]...)
    end

    expK = UR * exp(W) * UL

    function exp_pushback(f̄wd)
        ēxpK = f̄wd 
       
        K̄ = zero(K)

        if ēxpK != ZeroTangent()
            if W.data isa TensorKit.SortedVectorDict
                # TODO. symmetric tensor
                error("symmetric tensor. not implemented")
            end
            function coeff(a::Number, b::Number) 
                if a ≈ b
                    return exp(a)
                else 
                    return (exp(a) - exp(b)) / (a - b)
                end
            end
            M = UR' * ēxpK * UL'
            M1 = similar(M)
            copyto!(M1.data, M.data .* coeff.(Ws', conj.(Ws)))
            K̄ += UL' * M1 * UR'# - tr(ēxpK * expK') * expK'
        end
        return NoTangent(), K̄
    end 
    return expK, exp_pushback
end