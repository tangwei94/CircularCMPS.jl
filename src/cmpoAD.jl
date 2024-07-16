function otimes_rev_l(X::AbstractTensorMap{S, 2, 2}, B::AbstractTensorMap{S, 1, 1}) where S
    @tensor rev[-1; -2] := X[-1 1; -2 2] * B'[2; 1]
    return rev
end
function otimes_rev_r(X::AbstractTensorMap{S, 2, 2}, A::AbstractTensorMap{S, 1, 1}) where S
    @tensor rev[-1; -2] := X[1 -1; 2 -2] * A'[2; 1]
    return rev
end

function ChainRulesCore.rrule(::typeof(TensorKit.:⊗), A::MPSBondTensor, B::MPSBondTensor)
    function otimes_pushback(f̄wd)
        Ā = otimes_rev_l(f̄wd, B)
        B̄ = otimes_rev_r(f̄wd, A)
        return NoTangent(), Ā, B̄
    end
    return A⊗B, otimes_pushback
end

function ChainRulesCore.rrule(::typeof(Base.:*), W::Matrix{<:Number}, Rs::Vector{<:MPSBondTensor})
    function times_pushback(f̄wd)
        W̄ = zeros(eltype(W), size(W))
        for ix in 1:size(W)[1], iy in 1:size(W)[2]
            W̄[ix, iy] = tr(f̄wd[ix] * Rs[iy]')
        end
        return NoTangent(), W̄, NoTangent()
    end
    return W*Rs, times_pushback
end
function Base.:*(Rs::Vector{<:MPSBondTensor}, W::Matrix{<:Number}) 
    Rs1 = Vector{MPSBondTensor}(undef, length(Rs))
    for ix in eachindex(Rs)
        Rs1[ix] = sum(Rs .* W[:, ix])
    end
    return Rs1
end
function ChainRulesCore.rrule(::typeof(Base.:*), Rs::Vector{<:MPSBondTensor}, W::Matrix{<:Number})
    function times_pushback(f̄wd)
        W̄ = zeros(eltype(W), size(W))
        for ix in 1:size(W)[1], iy in 1:size(W)[2]
            W̄[ix, iy] = tr(f̄wd[iy] * Rs[ix]')
        end
        return NoTangent(), NoTangent(), W̄
    end
    return Rs*W, times_pushback
end
function ChainRulesCore.rrule(::typeof(Base.:*), W::Matrix{<:Number}, Ps::Matrix{<:MPSBondTensor})
    function times_pushback(f̄wd)
        W̄ = zeros(eltype(W), size(W))
        for ix in 1:size(W)[1], iy in 1:size(W)[2]
            W̄[ix, iy] = tr(sum(f̄wd[ix, :] * Ps[iy, :]'))
        end
        return NoTangent(), W̄, NoTangent()
    end
    return W*Ps, times_pushback
end
function ChainRulesCore.rrule(::typeof(Base.:*), Ps::Matrix{<:MPSBondTensor}, W::Matrix{<:Number})
    function times_pushback(f̄wd)
        W̄ = zeros(eltype(W), size(W))
        for ix in 1:size(W)[1], iy in 1:size(W)[2]
            W̄[ix, iy] = tr(sum(f̄wd[:, iy] * Ps[:, ix]'))
        end
        return NoTangent(), NoTangent(), W̄
    end
    return Ps*W, times_pushback
end

function ChainRulesCore.rrule(::typeof(Base.:*), M::Matrix{<:AbstractTensorMap}, v::Vector{<:AbstractTensorMap})
    Mv = [sum(M[ix, :] .⊗ v) for ix in eachindex(v)]
    sizev = size(M)[1]
    function times_pushback(f̄wd)
        v̄ = Vector{AbstractTensorMap}(undef, sizev)
        M̄ = Matrix{AbstractTensorMap}(undef, sizev, sizev)

        for iy in 1:sizev 
            v̄[iy] = sum(otimes_rev_r.(f̄wd, M[:, iy]))
        end
        for ix in 1:sizev, iy in 1:sizev
            M̄[ix, iy] = otimes_rev_l(f̄wd[ix], v[iy])
        end
        return NoTangent(), M̄, v̄
    end
    return Mv, times_pushback
end
function ChainRulesCore.rrule(::typeof(Base.:*), v::Vector{<:AbstractTensorMap}, M::Matrix{<:AbstractTensorMap})
    vM = [sum(v .⊗ M[:, ix]) for ix in eachindex(v)]
    sizev = size(M)[1]
    function times_pushback(f̄wd)
        v̄ = Vector{AbstractTensorMap}(undef, sizev)
        M̄ = Matrix{AbstractTensorMap}(undef, sizev, sizev)

        for iy in 1:sizev 
            v̄[iy] = sum(otimes_rev_l.(f̄wd, M[iy, :]))
        end
        for ix in 1:sizev, iy in 1:sizev
            M̄[iy, ix] = otimes_rev_r(f̄wd[ix], v[iy])
        end
        return NoTangent(), v̄, M̄
    end
    return vM, times_pushback
end
function ChainRulesCore.rrule(::typeof(Base.:*), M::Matrix{<:AbstractTensorMap}, N::Matrix{<:AbstractTensorMap})
    sizev = size(M)[1]
    MN = [sum(M[ix, :] .⊗ N[:, iy]) for ix in 1:sizev, iy in 1:sizev]
    function times_pushback(f̄wd)
        M̄ = Matrix{AbstractTensorMap}(undef, sizev, sizev)
        N̄ = Matrix{AbstractTensorMap}(undef, sizev, sizev)

        for ix in 1:sizev, iy in 1:sizev
            M̄[ix, iy] = sum(otimes_rev_l(f̄wd[ix, :], N[iy, :]))
            N̄[ix, iy] = sum(otimes_rev_r.(f̄wd[:, iy], M[:, ix]))
        end
        return NoTangent(), M̄, N̄
    end
    return MN, times_pushback
end