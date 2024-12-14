@testset "test basic utility functions for MultiBosonCMPSData_tnp" for ix in 1:10
    ψa = MultiBosonCMPSData_tnp(rand, 2, 3)

    @test norm(ψa) / norm(2*ψa) ≈ 0.5
    @test norm(ψa) / norm(ψa + ψa * 0.5) ≈ 2/3
    @test norm(ψa) / norm(ψa - ψa * 0.5) ≈ 2

    ψb = similar(ψa)
    randomize!(ψb)
    @test get_χ(ψb) == 8 
    @test get_d(ψb) == 3 

end

@testset "test MultiBosonCMPSData_tnp to CMPSData conversion" for ix in 1:10
    χb, d = 3, 2
    χ = χb^d
    ψ = MultiBosonCMPSData_tnp(rand, χb, d)
    ϕn = CMPSData(rand, χ, d)

    ψn = CMPSData(ψ)
    #@test norm(ψn - CMPSData(MultiBosonCMPSData_tnp(ψn))) < 1e-12

    function _F1(ψ)
        ψn = CMPSData(ψ)

        TM1 = TransferMatrix(ϕn, ψn)
        TM2 = TransferMatrix(ψn, ϕn)
        vl1 = left_env(TM1)
        vr2 = right_env(TM2)
   
        return norm(tr(vl1)) / norm(vl1) + norm(tr(vr2))/norm(vr2) + norm(tr(vl1 * vr2)) / norm(vl1) / norm(vr2)
    end
    function _F2(ψ)
        ψn = CMPSData(ψ)

        TM1 = TransferMatrix(ϕn, ψn)
        TM2 = TransferMatrix(ψn, ψn)
        vl1 = left_env(TM1)
        vr2 = right_env(TM2)
   
        return norm(tr(vl1)) / norm(vl1) + norm(tr(vr2)) / norm(vr2) + norm(tr(vl1 * vr2)) / norm(vl1) / norm(vr2)
    end
    cs, μs = ComplexF64[1. 1.4; 1.4 2.], ComplexF64[2.1, 2.3]
    function _FE(ψ::MultiBosonCMPSData_tnp)
        ψn = CMPSData(ψ)
        OH = kinetic(ψn) + cs[1,1]*point_interaction(ψn, 1) + cs[2,2]*point_interaction(ψn, 2) + cs[1,2] * point_interaction(ψn, 1, 2) + cs[2,1] * point_interaction(ψn, 2, 1) - μs[1] * particle_density(ψn, 1) - μs[2] * particle_density(ψn, 2)
        TM = TransferMatrix(ψn, ψn)
        envL = permute(left_env(TM), ((), (1, 2)))
        envR = permute(right_env(TM), ((2, 1), ())) 
        return real(tr(envL * OH * envR) / tr(envL * envR))
    end

    test_ADgrad(_F1, ψ)
    test_ADgrad(_F2, ψ)
    test_ADgrad(_FE, ψ)
end

@testset "left canonical form MultiBosonCMPSData_tnp" for ix in 1:10
    χb, d = 3, 2
    χ = χb^d
    ψ = MultiBosonCMPSData_tnp(rand, χb, d)
    ψl = left_canonical(ψ)

    function check_left_canonical_form(x)
        cmps = CMPSData(x)
        @test norm(cmps.Q + cmps.Q' + sum([R' * R for R in cmps.Rs])) < 1e-9
    end

    check_left_canonical_form(ψl)

    α = 0.1*rand()
    dBs = [rand(ComplexF64, χb, χb) for ix in 1:d]
    X = randn(ComplexF64, χ, χ)
    ψl1 = CircularCMPS.retract_left_canonical(ψl, α, dBs, X)
    
    check_left_canonical_form(ψl1)

end

@testset "test diff_to_grad" for ix in 1:10
    χb, d = 3, 2
    χ = χb^2
    ψ = MultiBosonCMPSData_tnp(rand, χb, d)
    ψ = left_canonical(ψ)
    Cs = CircularCMPS.construct_full_block_matrix(ψ.Bs)
    Rs = Ref(ψ.M) .* Cs .* Ref(ψ.Minv)

    cs, μs = ComplexF64[1. 1.4; 1.4 2.], ComplexF64[2.1, 2.3]
    function _F1(ψ::MultiBosonCMPSData_tnp)
        ψn = CMPSData(ψ)
        OH = kinetic(ψn) + cs[1,1]*point_interaction(ψn, 1) + cs[2,2]*point_interaction(ψn, 2) + cs[1,2] * point_interaction(ψn, 1, 2) + cs[2,1] * point_interaction(ψn, 2, 1) - μs[1] * particle_density(ψn, 1) - μs[2] * particle_density(ψn, 2)
        TM = TransferMatrix(ψn, ψn)
        envL = permute(left_env(TM), ((), (1, 2)))
        envR = permute(right_env(TM), ((2, 1), ())) 
        return real(tr(envL * OH * envR) / tr(envL * envR))
    end
    ∂ψ = _F1'(ψ)
    ∂Bs = ∂ψ.Bs
    g0 = CircularCMPS.diff_to_grad(ψ, ∂ψ)

    function tangent_vec(g::MultiBosonCMPSData_tnp_Grad)
        dCs = CircularCMPS.construct_full_block_matrix(g.dBs)
        Ws = Ref(ψ.M) .* [g.X * C - C * g.X + dC for (C, dC) in zip(Cs, dCs)] .* Ref(ψ.Minv)
        #Ws = [ψ.M * g.X * ψ.Minv * R - R * ψ.M * g.X * ψ.Minv + ψ.M * dD * ψ.Minv for (R, dD) in zip(Rs, g.dDs)]
        V = - sum([R' * W for (R, W) in zip(Rs, Ws)])
        gM = ψ.M * g.X 
        return (V, g.dBs, gM)
    end

    # check the implementation
    g1 = similar(g0)
    randomize!(g1)
    rQ1, rBs1, rM1 = tangent_vec(g1)
    @test dot(g0, g1) ≈ dot(∂ψ.Q, rQ1) + sum(dot.(∂Bs, rBs1)) + dot(∂ψ.M, rM1)

    # check in the context of retraction
    α = 1e-6
    ψ2 = CircularCMPS.retract_left_canonical(ψ, α, g1.dBs, g1.X)
    ψ1 = CircularCMPS.retract_left_canonical(ψ, -α, g1.dBs, g1.X)
    @test norm((_F1(ψ2) - _F1(ψ1)) / (2*α) - real(dot(g0, g1))) / norm(dot(g0, g1)) < 1e-6
end

@testset "tangent_map" for ix in 1:10
    χb, d = 3, 2
    χ = χb^d
    ψ = MultiBosonCMPSData_tnp(rand, χb, d)
    ψ = left_canonical(ψ)
    ρR = right_env(ψ)
    Rs = Ref(ψ.M) .* CircularCMPS.construct_full_block_matrix(ψ.Bs) .* Ref(ψ.Minv)
    ϕn = CMPSData(rand, χ, d)

    function _F1(ψ)
        ψn = CMPSData(ψ)

        TM1 = TransferMatrix(ϕn, ψn)
        TM2 = TransferMatrix(ψn, ϕn)
        vl1 = left_env(TM1)
        vr2 = right_env(TM2)
   
        return norm(tr(vl1)) / norm(vl1) + norm(tr(vr2))/norm(vr2) + norm(tr(vl1 * vr2)) / norm(vl1) / norm(vr2)
    end
    ∂ψ = _F1'(ψ)
    ∂Bs = ∂ψ.Bs
    g0 = CircularCMPS.diff_to_grad(ψ, ∂ψ)

    g1 = similar(g0)
    randomize!(g1)
    g2 = similar(g0)
    randomize!(g2)

    function tangent_vec(g::MultiBosonCMPSData_tnp_Grad)
        X = g.X
        Cs = CircularCMPS.construct_full_block_matrix(ψ.Bs)
        dCs = CircularCMPS.construct_full_block_matrix(g.dBs)
        Ws = Ref(ψ.M) .* [X * C - C * X + dC for (C, dC) in zip(Cs, dCs)] .* Ref(ψ.Minv)
        V = - sum([R' * W for (R, W) in zip(Rs, Ws)])
        return (V, Ws)
    end

    V1, W1s = tangent_vec(g1)
    V2, W2s = tangent_vec(g2)

    ovlp1 = sum([tr(W1 * ρR * W2') for (W1, W2) in zip(W1s, W2s)])
    ovlp2 = dot(g2, tangent_map(ψ, g1; ρR=ρR)) 
    @test ovlp1 ≈ ovlp2

    ovlp1 = sum([tr(W1 * ρR * W1') for (W1, W2) in zip(W1s, W2s)])
    ovlp2 = dot(g1, tangent_map(ψ, g1; ρR=ρR)) 
    @test ovlp1 ≈ ovlp2
    @test norm(imag(ovlp1)) < 1e-12
end

@testset "tangent_map should be hermitian" for ix in 1:10
    χb, d = 3, 2
    χ = χb^2
    ψ = MultiBosonCMPSData_tnp(rand, χb, d)

    M = zeros(ComplexF64, d*(χb^2)+(χ^2), d*(χb^2)+(χ^2))
    for ix in 1:d*(χb^2)+(χ^2)
        v = zeros(ComplexF64, d*(χb^2)+(χ^2))
        v[ix] = 1
        g = MultiBosonCMPSData_tnp_Grad(v, χb, d)
        M[:, ix] = vec(tangent_map(ψ, g))
    end

    @test norm(M - M') < 1e-12

    Λ, _ = eigen(Hermitian(M))
    @test all(Λ .≥ -1e-14)

end