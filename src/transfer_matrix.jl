abstract type AbstractTransferMatrix end

struct TransferMatrix <: AbstractTransferMatrix
    Qu::MPSBondTensor
    Qd::MPSBondTensor
    Rus::Vector{<:MPSBondTensor}
    Rds::Vector{<:MPSBondTensor}
    is_flipped::Bool
end

function TransferMatrix(ψu::CMPSData, ψd::CMPSData)
    Qu, Qd = ψu.Q, ψd.Q
    Rus, Rds = ψu.Rs, ψd.Rs
    return TransferMatrix(Qu, Qd, Rus, Rds, false)
end

function TensorKit.flip(TM::TransferMatrix)
    return TransferMatrix(TM.Qu, TM.Qd, TM.Rus, TM.Rds, true)
end

function (TM::TransferMatrix)(v::MPSBondTensor)
    
    if TM.is_flipped == false # right eigenvector
        Tv = TM.Qd * v + v * TM.Qu' 
        for (Rd, Ru) in zip(TM.Rds, TM.Rus)
            Tv += Rd * v * Ru'
        end
        return Tv
    else # right eigenvector
        Tv = v * TM.Qd + TM.Qu' * v 
        for (Rd, Ru) in zip(TM.Rds, TM.Rus)
            Tv += Ru' * v * Rd
        end
        return Tv
    end
end

function right_env(TM::TransferMatrix)
    init = similar(TM.Qu, _firstspace(TM.Qd)←_firstspace(TM.Qu))
    randomize!(init);
    _, vrs, _ = eigsolve(TM, init, 1, :LR)
    return vrs[1]
end

function left_env(TM::TransferMatrix)
    init = similar(TM.Qu, _firstspace(TM.Qu)←_firstspace(TM.Qd))
    randomize!(init);
    _, vrs, _ = eigsolve(flip(TM), init, 1, :LR)
    return vrs[1]
end

# backward for right_env, left_env, TransferMatrix
struct TransferMatrixBackward <: AbstractTransferMatrix
    VLs::Vector{<:MPSBondTensor}
    VRs::Vector{<:MPSBondTensor}
end

Base.:+(bTM1::TransferMatrixBackward, bTM2::TransferMatrixBackward) = TransferMatrixBackward([bTM1.VLs; bTM2.VLs], [bTM1.VRs; bTM2.VRs])
Base.:-(bTM1::TransferMatrixBackward, bTM2::TransferMatrixBackward) = TransferMatrixBackward([bTM1.VLs; bTM2.VLs], [bTM1.VRs; -1*bTM2.VRs])
Base.:*(a::Number, bTM::TransferMatrixBackward) = TransferMatrixBackward(bTM.VLs, a * bTM.VRs)
Base.:*(bTM::TransferMatrixBackward, a::Number) = TransferMatrixBackward(bTM.VLs, a * bTM.VRs)

function (bTM::TransferMatrixBackward)(v::MPSBondTensor, cavity_loc::Symbol)
    if cavity_loc == :B
        bwd = zero(similar(v, codomain(bTM.VLs[1]) ← domain(bTM.VRs[1])))
        for (VL, VR) in zip(bTM.VLs, bTM.VRs) 
            bwd += VL * v * VR
        end
        return bwd 
    end

    if cavity_loc == :U 
        bwd = zero(similar(v, domain(bTM.VLs[1]) ← codomain(bTM.VRs[1])))
        for (VL, VR) in zip(bTM.VLs, bTM.VRs)
            bwd += VL' * v * VR'
        end
        return bwd        
    end
end

function right_env_backward(TM::TransferMatrix, λ::Number, vr::MPSBondTensor, ∂vr::MPSBondTensor)
    init = similar(vr)
    randomize!(init); 
    init = init - dot(vr, init) * vr # important. the subtracted part lives in the null space of flip(TM) - λ*I
    
    (norm(dot(vr, ∂vr)) > 1e-9) && @warn "right_env_backward: forward computation not gauge invariant: final computation should not depend on the phase of vr." # important
    #∂vr = ∂vr - dot(vr, ∂vr) * vr 
    ξr_adj, info = linsolve(x -> flip(TM)(x) - λ*x, ∂vr', init'; maxiter=500) # subtle
    (info.converged == 0) && @warn "right_env_backward not converged: normres = $(info.normres)"
    
    return ξr_adj'
end

function left_env_backward(TM::TransferMatrix, λ::Number, vl::MPSBondTensor, ∂vl::MPSBondTensor)
    init = similar(vl)
    randomize!(init); 
    init = init - dot(vl, init) * vl # important

    (norm(dot(vl, ∂vl)) > 1e-9) && @warn "left_env_backward: forward computation not gauge invariant: final computation should not depend on the phase of vl." # important
    ξl_adj, info = linsolve(x -> TM(x) - λ*x, ∂vl', init'; maxiter=500) # subtle
    (info.converged == 0) && @warn "left_env_backward not converged: normres = $(info.normres)"

    return ξl_adj'
end

function ChainRulesCore.rrule(::typeof(right_env), TM::TransferMatrix)
    init = similar(TM.Qu, _firstspace(TM.Qd)←_firstspace(TM.Qu))
    randomize!(init);
    λrs, vrs, _ = eigsolve(TM, init, 1, :LR)
    λr, vr = λrs[1], vrs[1]
    
    function right_env_pushback(∂vr)
        ξr = right_env_backward(TM, λr, vr, ∂vr)
        return NoTangent(), TransferMatrixBackward([-ξr], [vr'])
    end
    return vr, right_env_pushback
end

function ChainRulesCore.rrule(::typeof(left_env), TM::TransferMatrix)
    init = similar(TM.Qu, _firstspace(TM.Qu)←_firstspace(TM.Qd))
    randomize!(init);
    λls, vls, _ = eigsolve(flip(TM), init, 1, :LR)
    λl, vl = λls[1], vls[1]
   
    function left_env_pushback(∂vl)
        ξl = left_env_backward(TM, λl, vl, ∂vl)
        return NoTangent(), TransferMatrixBackward([vl'], [-ξl])
    end
    return vl, left_env_pushback
end

function ChainRulesCore.rrule(::Type{TransferMatrix}, ψu::CMPSData, ψd::CMPSData)
    Qu, Qd = ψu.Q, ψd.Q
    Rus, Rds = ψu.Rs, ψd.Rs
    TM = TransferMatrix(Qu, Qd, Rus, Rds, false)

    function TransferMatrix_pushback(∂TM)
        ∂Qd = ∂TM(id(_firstspace(Qu)), :B)
        ∂Rds = [∂TM(Ru, :B) for Ru in Rus]
        
        ∂Qu = ∂TM(id(_firstspace(Qd)), :U)
        ∂Rus = [∂TM(Rd, :U) for Rd in Rds]

        return NoTangent(), CMPSData(∂Qu, ∂Rus), CMPSData(∂Qd, ∂Rds)
    end
    return TM, TransferMatrix_pushback 
end


function right_env(K::AbstractTensorMap{T, S, 2, 2}) where {T,S} 
    init = similar(K, _firstspace(K)←_lastspace(K)')
    randomize!(init);
    λrs, vrs, _ = eigsolve(v -> Kact_R(K, v), init, 1, :LR)
    return λrs[1], vrs[1]
end

function left_env(K::AbstractTensorMap{T, S, 2, 2}) where {T,S}
    init = similar(K, _lastspace(K)'←_firstspace(K))
    randomize!(init);
    λls, vls, _ = eigsolve(v -> Kact_L(K, v), init, 1, :LR)
    return λls[1], vls[1]
end

function Kmat_pseudo_inv(K::AbstractTensorMap{T, S, 2, 2}, λ::Number) where {T,S}
    χ = dim(_firstspace(K))
    IdK = id((ℂ^χ)'⊗ℂ^χ)
    K1 = K_permute_back(K) - λ * IdK # normalize

    Λ, U = eigen(K1)
    Λ.data[end] = 0 # I add this to avoid numerical instablity
    function f_pseudo_inv(x::Number)
        if norm(x) < 1e-12
            return x
        else
            return 1/x
        end
    end
    for (k,v) in blocks(Λ)
        map!(x->f_pseudo_inv(x), v, v);
    end

    Kinv = K_permute(U * Λ * inv(U)) 

    return -Kinv
end
