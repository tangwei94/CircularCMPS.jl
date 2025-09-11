function measure_local_observable(ψ::CMPSData, O::AbstractTensorMap, L::Real)
    if L == Inf
        TM = TransferMatrix(ψ, ψ)
        ρR = permute(right_env(TM), ((2, 1), ()))
        ρL = permute(left_env(TM), ((), (1, 2)))

        return tr(ρL * O * ρR) / tr(ρL * ρR)
    else
        error("Not implemented")
    end
end

function correlator(ψ::CMPSData, L::Real)
    if L == Inf
        TM = TransferMatrix(ψ, ψ)
        ρR = permute(right_env(TM), ((2, 1), ()))
        ρL = permute(left_env(TM), ((), (1, 2)))

        Kmat = K_mat(ψ, ψ)
        Λs, V = eig(Kmat)
        Vinv = inv(V)

        function corr_inf(O1::AbstractTensorMap, O2::AbstractTensorMap, Δx::Real)
            return tr(ρL * O1 * V * exp(Λs*Δx) * Vinv * O2 * ρR) / tr(ρL * ρR)
        end
        
        TM_spectra = Λs.data
        order_TM_spectra = sortperm(real.(TM_spectra))
        TM_spectra = TM_spectra[order_TM_spectra]

        return corr_inf, TM_spectra
    else
        Kmat = K_mat(ψ, ψ)
        Λs, V = eig(Kmat)
        Vinv = inv(V)
        
        function corr_L(O1::AbstractTensorMap, O2::AbstractTensorMap, Δx::Real)
            return tr(Vinv * O1 * V * exp(Λs*Δx) * Vinv * O2 * V * exp(Λs*(L-Δx)))
        end

        TM_spectra = Λs.data
        order_TM_spectra = sortperm(real.(TM_spectra))
        TM_spectra = TM_spectra[order_TM_spectra]

        return corr_inf, TM_spectra
    end
end