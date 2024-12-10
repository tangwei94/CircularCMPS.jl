@testset "test basic utility functions for MultiBosonCMPSData_MDMinv" for ix in 1:10
    ψa = MultiBosonCMPSData_MDMinv(rand, 2, 3)

    @test norm(ψa) / norm(2*ψa) ≈ 0.5
    @test norm(ψa) / norm(ψa + ψa * 0.5) ≈ 2/3
    @test norm(ψa) / norm(ψa - ψa * 0.5) ≈ 2

    ψb = similar(ψa)
    randomize!(ψb)
    @test get_χ(ψb) == 2 
    @test get_d(ψb) == 3 

end

@testset "test MultiBosonCMPSData_MDMinv to CMPSData conversion" for ix in 1:10
    χ, d = 4, 2
    ψ = MultiBosonCMPSData_MDMinv(rand, χ, d)
    ϕn = CMPSData(rand, χ, d)

    ψn = CMPSData(ψ)
    @test norm(ψn - CMPSData(MultiBosonCMPSData_MDMinv(ψn))) < 1e-12

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
   
        return norm(tr(vl1)) / norm(vl1) + norm(tr(vr2))/norm(vr2) + norm(tr(vl1 * vr2)) / norm(vl1) / norm(vr2)
    end
    cs, μs = ComplexF64[1. 1.4; 1.4 2.], ComplexF64[2.1, 2.3]
    function _FE(ψ::MultiBosonCMPSData_MDMinv)
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

@testset "left canonical form MultiBosonCMPSData_MDMinv" for ix in 1:10
    χ, d = 4, 2
    ψ = MultiBosonCMPSData_MDMinv(rand, χ, d)
    ψl = left_canonical(ψ)

    function check_left_canonical_form(x)
        cmps = CMPSData(x)
        @test norm(cmps.Q + cmps.Q' + sum([R' * R for R in cmps.Rs])) < 1e-11
    end

    check_left_canonical_form(ψl)

    α = rand()
    dDs = [Diagonal(randn(ComplexF64, χ)) for ix in 1:d]
    X = randn(ComplexF64, χ, χ)
    ψl1 = CircularCMPS.retract_left_canonical(ψl, α, dDs, X)
    
    check_left_canonical_form(ψl1)

end

@testset "test diff_to_grad" for ix in 1:10
    χ, d = 4, 2
    ψ = MultiBosonCMPSData_MDMinv(rand, χ, d)
    ψ = left_canonical(ψ)
    Rs = Ref(ψ.M) .* ψ.Ds .* Ref(ψ.Minv)

    cs, μs = ComplexF64[1. 1.4; 1.4 2.], ComplexF64[2.1, 2.3]
    function _F1(ψ::MultiBosonCMPSData_MDMinv)
        ψn = CMPSData(ψ)
        OH = kinetic(ψn) + cs[1,1]*point_interaction(ψn, 1) + cs[2,2]*point_interaction(ψn, 2) + cs[1,2] * point_interaction(ψn, 1, 2) + cs[2,1] * point_interaction(ψn, 2, 1) - μs[1] * particle_density(ψn, 1) - μs[2] * particle_density(ψn, 2)
        TM = TransferMatrix(ψn, ψn)
        envL = permute(left_env(TM), ((), (1, 2)))
        envR = permute(right_env(TM), ((2, 1), ())) 
        return real(tr(envL * OH * envR) / tr(envL * envR))
    end
    ∂ψ = _F1'(ψ)
    ∂Ds = ∂ψ.Ds
    g0 = CircularCMPS.diff_to_grad(ψ, ∂ψ)

    function tangent_vec(g::MultiBosonCMPSData_MDMinv_Grad)
        Ws = [g.X * R - R * g.X + ψ.M * dD * ψ.Minv for (R, dD) in zip(Rs, g.dDs)]
        V = - sum([R' * W for (R, W) in zip(Rs, Ws)])
        gM = g.X * ψ.M
        return (V, g.dDs, gM)
    end

    # check the implementation
    g1 = similar(g0)
    randomize!(g1)
    rQ1, rDs1, rM1 = tangent_vec(g1)
    @test dot(g0, g1) ≈ dot(∂ψ.Q, rQ1) + sum(dot.(∂Ds, rDs1)) + dot(∂ψ.M, rM1)

    # check in the context of retraction
    α = 1e-4
    ψ2 = CircularCMPS.retract_left_canonical(ψ, α, g1.dDs, g1.X)
    ψ1 = CircularCMPS.retract_left_canonical(ψ, -α, g1.dDs, g1.X)
    @test norm((_F1(ψ2) - _F1(ψ1)) / (2*α) - real(dot(g0, g1))) < 1e-4
end

#@testset "test Kmat_pseudo_inv by numerical integration" for ix in 1:10
#    χ, d = 4, 2
#    ψ = MultiBosonCMPSData_P(rand, χ, d)
#
#    ψn = CMPSData(ψ);
#    K = K_permute(K_mat(ψn, ψn));
#    λ, EL = left_env(K);
#    λ, ER = right_env(K);
#    Kinv = Kmat_pseudo_inv(K, λ);
#
#    VL = Tensor(rand, ComplexF64, (ℂ^(χ^d))'⊗ℂ^(χ^d))
#    VR = Tensor(rand, ComplexF64, (ℂ^(χ^d))'⊗ℂ^(χ^d))
#
#    IdK = K_permute(id((ℂ^(χ^d))'⊗ℂ^(χ^d)))
#    K_nm = K - λ * IdK
#    K0_nm = K_permute_back(K_nm)
#
#    Kinf = exp(1e4*K0_nm)
#    @test norm(Kinf) ≈ 1/norm(tr(EL * ER))
#
#    Λ0, U0 = eigen(K0_nm)
#    @test norm(exp(12*K0_nm) - U0 * exp(12*Λ0) * inv(U0)) < 1e-12
#    
#    δ = isometry(ℂ^(χ^(2*d)), ℂ^(χ^(2*d)-1))
#    Λr, Ur, invUr = δ' * Λ0 * δ, U0 * δ, δ' * inv(U0)
#
#    a1, err = quadgk(τ -> tr(VL' * Ur * exp(τ * Λr) * invUr * VR), 0, 1e4)
#    a2 = tr(VL' * K_permute_back(Kinv) * VR)
#    @test norm(a1 - a2) < 100 * err
#
#end

@testset "tangent_map" for ix in 1:10
    χ, d = 4, 2
    ψ = MultiBosonCMPSData_MDMinv(rand, χ, d)
    ψ = left_canonical(ψ)
    ρR = right_env(ψ)
    Rs = Ref(ψ.M) .* ψ.Ds .* Ref(ψ.Minv)
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
    ∂Ds = ∂ψ.Ds
    g0 = CircularCMPS.diff_to_grad(ψ, ∂ψ)

    g1 = similar(g0)
    randomize!(g1)
    g2 = similar(g0)
    randomize!(g2)

    function tangent_vec(g::MultiBosonCMPSData_MDMinv_Grad)
        Ws = [g.X * R - R * g.X + ψ.M * dD * ψ.Minv for (R, dD) in zip(Rs, g.dDs)]
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
    χ, d = 4, 2
    ψ = MultiBosonCMPSData_MDMinv(rand, χ, d)

    M = zeros(ComplexF64, χ*d+χ^2, χ*d+χ^2)
    for ix in 1:χ*d+χ^2
        v = zeros(ComplexF64, χ*d+χ^2)
        v[ix] = 1
        g = MultiBosonCMPSData_MDMinv_Grad(v, χ, d)
        M[:, ix] = vec(tangent_map(ψ, g))
    end

    @test norm(M - M') < 1e-12

    Λ, _ = eigen(Hermitian(M))
    @test all(Λ .≥ -1e-14)

end