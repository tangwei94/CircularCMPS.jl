@testset "test gradient for MultiBosonCMPSData_MCMinv -> CMPSData" for ix in 1:10
    Q = rand(ComplexF64, 4, 4)
    M = rand(ComplexF64, 4, 4)
    Cs = map(ix -> rand(ComplexF64, 4, 4), 1:2)
    α = 0.1

    ψ_mcm = MultiBosonCMPSData_MCMinv(Q, M, Cs, α)
    ψ = CMPSData(ψ_mcm)

    ϕn = CMPSData(rand, 4, 2)
    function _F1(ψm)
        ψn = CMPSData(ψm)

        TM1 = TransferMatrix(ϕn, ψn)
        TM2 = TransferMatrix(ψn, ϕn)
        vl1 = left_env(TM1)
        vr2 = right_env(TM2)

        return norm(tr(vl1)) / norm(vl1) + norm(tr(vr2))/norm(vr2) + norm(tr(vl1 * vr2)) / norm(vl1) / norm(vr2)
    end

    _F1(ψ_mcm)
    v, g = withgradient(_F1, ψ_mcm)

    ψd = MultiBosonCMPSData_MCMinv(rand, 4, 2, 1.0)
    ψd = ψd * (1/ sqrt(dot(ψd, ψd)))

    ϵ = 1e-4
    ψ_mcm + ϵ * ψd
    Fu = _F1(ψ_mcm + ϵ * ψd)
    Fd = _F1(ψ_mcm - ϵ * ψd)

    @test (Fu - Fd) / (2 * ϵ) ≈ real(dot(ψd, g[1]))
end