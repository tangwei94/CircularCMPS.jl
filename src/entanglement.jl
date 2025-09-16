function entanglement_spectrum_inf(ψ::CMPSData)
    TM = TransferMatrix(ψ, ψ)
    ρR = right_env(TM)
    ρL = left_env(TM)
    ρ = ρL * ρR
    
    _, S, _ = tsvd(ρ)
    return S.data / tr(S)
end

function entanglement_entropy_inf(ψ::CMPSData)
    S = entanglement_spectrum_inf(ψ)
    return - real(sum(S .* log.(S)))
end


function half_chain_singular_values(ψ::CMPSData, β::Real)
    #MA = permute(finite_env(ψ*ψ, β/2)[1], (1, 3), (2, 4))
    #MB = permute(finite_env(convert(TensorMap, (ψ*ψ)'), β/2)[1], (1, 3), (2, 4))

    #ρA = sqrt(MA) * MB * sqrt(MA)
    #ρA = ρA / tr(ρA)

    #ΛA, _ = eigen(ρA)
    #return ΛA
    return half_chain_singular_values(ψ, ψ, β)
end

function half_chain_singular_values(ψL::CMPSData, ψR::CMPSData, β::Real)
    MA = permute(finite_env(ψL*ψR, β/2)[1], (1, 3), (2, 4))
    #MB = permute(finite_env(convert(TensorMap, (ψL*ψR)'), β/2)[1], (1, 3), (2, 4))
    MB = permute(finite_env(ψL*ψR, β/2)[1], (4, 2), (3, 1))

    ρA = sqrt(MA) * MB * sqrt(MA)
    ρA = ρA / tr(ρA)

    ΛA, _ = eigen(ρA)
    return ΛA
end

function half_chain_singular_values_testtool(ψL::CMPSData, ψR::CMPSData, β::Real)
    MLR = permute(finite_env(ψL*ψR, β/2)[1], (1, 3), (2, 4))
    MAR = permute(finite_env(ψR*ψR, β/2)[1], (1, 3), (2, 4))
    MBR = permute(finite_env(ψR*ψR, β/2)[1], (4, 2), (3, 1))
    MAL = permute(finite_env(ψL*ψL, β/2)[1], (1, 3), (2, 4))
    MBL = permute(finite_env(ψL*ψL, β/2)[1], (4, 2), (3, 1))

    ΛAR, UAR = eigh(MAR)
    CAR = sqrt(ΛAR) * UAR'
    ΛBR, UBR = eigh(MBR)
    CBR = UBR * sqrt(ΛBR)
    CR = CAR * CBR 
    UR, SR, _ = tsvd(CR)
    invCAR = inv(CAR) 
    
    ΛAL, UAL = eigh(MAL)
    CAL = sqrt(ΛAL) * UAL'
    ΛBL, UBL = eigh(MBL)
    CBL = UBL * sqrt(ΛBL)
    CL = CAL * CBL 
    UL, SL, _ = tsvd(CL)
    invCAL = inv(CAL) 
    
    normalized_MLR = UL' * invCAL' * MLR * invCAR * UR

    return normalized_MLR, SL * normalized_MLR * SR
end

function entanglement_entropy(ψ::CMPSData, β::Real)
    ΛA = half_chain_singular_values(ψ, β)
    SA = - real(tr(ΛA * log(ΛA)))

    return SA 
end

function entanglement_entropy(ψL::CMPSData, ψR::CMPSData, β::Real)
    ΛA = half_chain_singular_values(ψL, ψR, β)
    SA = - real(tr(ΛA * log(ΛA)))

    return SA 
end

function suggest_χ(Wmat::Matrix, ψ::CMPSData, β::Real; tol::Real=1e-9, maxχ::Int=32, minχ::Int=2)
    ΛA = half_chain_singular_values(ψ, β)
    SA = reverse(norm.(diag(ΛA.data)))

    ψL = Wmat * ψ
    ΛA_LR = half_chain_singular_values(ψL, ψ, β)
    SA_LR = reverse(norm.(diag(ΛA_LR.data)))

    χ1 = min(ceil(Int, sqrt(sum(SA .> tol))), maxχ)
    χ2 = min(ceil(Int, sqrt(sum(SA_LR .> tol))), maxχ)
    χ = max(χ1, χ2, minχ)

    return χ 
end