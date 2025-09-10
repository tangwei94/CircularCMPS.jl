function correlator(ψ::CMPSData, L::Real)
    if L == Inf
        ρR = right_env(ψ, ψ)
        ρL = left_env(ψ, ψ)

        Kmat = K_mat(ψ, ψ)
        Λs, V = eig(Kmat)
        Vinv = inv(V)

        function corr(O1::AbstractTensorMap, O2::AbstractTensorMap, Δx::Real)
            return ρL * O1 * Vinv * exp(Λs*Δx) * V * O2 * ρR
        end

        return corr
    else
        error("Not implemented")
    end
end