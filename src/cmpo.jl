abstract type AbstractCMPO end 

struct CMPO <: AbstractCMPO 
    Q::MPSBondTensor
    Ls::Vector{<:MPSBondTensor}
    Rs::Vector{<:MPSBondTensor}
    Ps::Matrix{<:MPSBondTensor}
end

@inline get_χ(T::CMPO) = dim(_firstspace(T.Q))
TensorKit.space(T::CMPO) = _firstspace(T.Q)

function Base.:*(M::Matrix{<:AbstractTensorMap}, v::Vector{<:AbstractTensorMap})
    Mv = [sum(M[ix, :] .⊗ v) for ix in eachindex(v)]
    return Mv
end

function Base.:*(v::Vector{<:AbstractTensorMap}, M::Matrix{<:AbstractTensorMap})
    vM = AbstractTensorMap[]
    for ix in eachindex(v)
        push!(vM, sum(v .⊗ M[:, ix]))
    end
    return vM
end

function Base.:*(M::Matrix{<:AbstractTensorMap}, N::Matrix{<:AbstractTensorMap})
    diml, dimr = size(M)[1], size(N)[2]
    MN = Matrix{AbstractTensorMap}(undef, diml, dimr)
    for ix in 1:diml, iy in 1:dimr
        MN[ix, iy] = sum(M[ix, :] .⊗ N[:, iy])
    end
    return MN
end

function outer(v1::Vector{<:AbstractTensorMap}, v2::Vector{<:AbstractTensorMap})
    diml, dimr = length(v1), length(v2)
    v1v2 = Matrix{AbstractTensorMap}(undef, diml, dimr)
    for ix in 1:diml, iy in 1:dimr
        v1v2[ix, iy] = v1[ix] ⊗ v2[iy]
    end
    return v1v2
end

#function Base.:*(v::Vector{<:AbstractTensorMap}, M::Matrix{<:AbstractTensorMap})
#    Mv = AbstractTensorMap[]
#    for ix in eachindex(v)
#        push!(Mv, sum(v .⊗ M[:, ix]))
#    end
#    return Mv
#end

"""
    Base.:*(T::CMPO, ψ::CMPSData)

    Act cMPO `T` on the cMPS `ψ`. The computation doesn't involve the length of the cMPS. 
"""
function Base.:*(T::CMPO, ψ::CMPSData)
    χψ, χT = get_χ(ψ), get_χ(T)
    χ = χT * χψ
    
    t_fuse = isomorphism(ℂ^χ, ℂ^χT*ℂ^χψ)
    Id_T = id(_firstspace(T.Q))
    Id_ψ = id(_firstspace(ψ.Q))

    Qa = sum(T.Rs .⊗ ψ.Rs) + Id_T ⊗ ψ.Q + T.Q ⊗ Id_ψ
    Ras = T.Ls .⊗ Ref(Id_ψ) + T.Ps * ψ.Rs
    
    Q = t_fuse * Qa * t_fuse'
    Rs = Ref(t_fuse) .* Ras .* Ref(t_fuse')
    return CMPSData(Q, Rs)
end

function Base.:*(T1::CMPO, T2::CMPO)

    χ1, χ2 = get_χ(T1), get_χ(T2)
    χ = χ1 * χ2
    
    t_fuse = isomorphism(ℂ^χ, ℂ^χ1*ℂ^χ2)
    Id1 = id(_firstspace(T1.Q))
    Id2 = id(_firstspace(T2.Q))

    Q = sum(T1.Rs .⊗ T2.Ls) + Id1 ⊗ T2.Q + T1.Q ⊗ Id2
    Ls = T1.Ls .⊗ Ref(Id2) + T1.Ps * T2.Ls 
    Rs = Ref(Id1) .⊗ T2.Rs + T1.Rs * T2.Ps 
    Ps = T1.Ps * T2.Ps
    
    Q = t_fuse * Q * t_fuse'
    Rs = Ref(t_fuse) .* Rs .* Ref(t_fuse')
    Ls = Ref(t_fuse) .* Ls .* Ref(t_fuse')
    Ps = Ref(t_fuse) .* Ps .* Ref(t_fuse')
    return CMPO(Q, Ls, Rs, Ps) 
end

#"""
#    Base.:*(ψ::CMPSData, T::CMPO)
#
#    Act cMPO `T` on the cMPS `ψ` from the left. The computation doesn't involve the length of the cMPS. 
#"""
#function Base.:*(ψ::CMPSData, T::CMPO)
#    χψ, χT = get_χ(ψ), get_χ(T)
#    χ = χT * χψ
#    
#    t_fuse = isomorphism(ℂ^χ, ℂ^χψ*ℂ^χT)
#    Id_T = id(_firstspace(T.Q))
#    Id_ψ = id(_firstspace(ψ.Q))
#
#    Q = sum(ψ.Rs .⊗ T.Ls) + Id_ψ ⊗ T.Q + ψ.Q ⊗ Id_T
#    Rs = ψ.Rs * T.Ps + Ref(Id_ψ) .⊗ T.Rs
#    
#    Q = t_fuse * Q * t_fuse'
#    Rs = Ref(t_fuse) .* Rs .* Ref(t_fuse')
#    return CMPSData(Q, Rs)
#
#end

"""
    Base.:*(ϕ::CMPSData, ψ::CMPSData)

    We define the product between two `CMPSData`'s as the `K_mat` obtained from the two cMPS's. 
"""
function Base.:*(ϕ::CMPSData, ψ::CMPSData)
    return K_mat(ϕ, ψ)
end

function ln_ovlp(ϕ::CMPSData, ψ::CMPSData, L::Real)
    return finite_env(ϕ*ψ, L)[2]
end

function ln_ovlp(ϕ::CMPSData, T::CMPO, ψ::CMPSData, L::Real)
    return finite_env(ϕ * (T * ψ), L)[2]
end

function compress(ψ::CMPSData, χ::Integer, L::Real; maxiter::Integer=100, tol::Real=1e-9, verbosity::Integer=1, init=nothing)

    if χ >= get_χ(ψ)
        @warn "no need to compress"
        return ψ
    end

    K = ψ * ψ
    expK, ln_norm = finite_env(K, L)

    # target function for variational optimization
    function _f(ϕ::CMPSData)
        return real(-ln_ovlp(ϕ, ψ, L) - ln_ovlp(ψ, ϕ, L) + ln_ovlp(ϕ, ϕ, L) + ln_norm)
    end

    # initial guess 1 ("reduced density matrix")
    @tensor M[-1; -2] := expK[1 -1 ; 1 -2]
    _, U = eig(M; sortby=λ->-real(λ))
    P = isometry(space(ψ), ℂ^χ)
    U1 = U * P
    U1inv = P' * inv(U) 

    Q1 = U1inv * ψ.Q * U1
    Rs1 = Ref(U1inv) .* ψ.Rs .* Ref(U1)
    ψ1 = CMPSData(Q1, Rs1)

    # initial guess 2 (pretend it's infinite. better for low T)
    #_, ψ2 = right_canonical(ψ);
    #CR, ψ2 = left_canonical(ψ2);
    #U2, _, _ = tsvd(CR, (1,), (2,); trunc= truncdim(χ))
    #Q2 = U2' * ψ.Q * U2
    #R2s = Ref(U2') .* ψ.Rs .* Ref(U2)
    #ψ2 = CMPSData(Q2, R2s)

    if init === nothing || get_χ(init) > χ 
        (verbosity >= 1) && println("no init available")
        init = ψ1#CMPSData(rand, χ, get_d(ψ1))
    end

    if get_χ(init) < χ
        init = expand(init, χ, L; perturb=tol)
    end

    #f1 = _f(ψ1)
    #f2 = _f(ψ2)
    #finit = _f(init)
    # prefers init
    #if 0.1*finit < f1 && 0.1*finit < f2 
    (verbosity >= 1) && printstyled("input init is chosen: finit: $(_f(init)), ginit: $(norm(_f'(init))) \n"; color=:light_black)
    #end
    #if f1 < f2 && f1 < 0.1*finit 
    #    (verbosity >= 1) && printstyled("input init is not chosen, use ψ1 instead. \n f1: $(f1), f2: $(f2), finit: $(finit) "; color=:light_black)
    #    init = ψ1
    #end
    #if f2 < f1 && f2 < 0.1*finit 
    #    (verbosity >= 1) && printstyled("input init is not chosen, use ψ2 instead. \n f1: $(f1), f2: $(f2), finit: $(finit) "; color=:light_black)
    #    init = ψ2
    #end
    
    # optimization 
    optalg = CircularCMPSRiemannian(maxiter, tol, verbosity)
    ψout, ln_fidel, grad, numfg, history = minimize(_f, init, optalg)

    return ψout#, ln_fidel, grad, numfg, history 
end

function leading_boundary_cmps(T::CMPO, init::CMPSData, β::Real; maxiter::Integer=1000, tol::Real=1e-9, verbosity::Integer=2, ϵ::Real=1e-6)

    function _f(ϕ::CMPSData)
        return -(1/β) * real(ln_ovlp(ϕ, T, ϕ, β) - ln_ovlp(ϕ, ϕ, β))
    end
    ψ1 = init + ϵ * CMPSData(rand, get_χ(init), get_d(init))
    
    # optimization 
    optalg = CircularCMPSRiemannian(maxiter, tol, verbosity)
    ψ1, f_result, grad, numfg, history = minimize(_f, ψ1, optalg)

    return ψ1, f_result, grad, numfg, history

end

function variance(T::CMPO, ψ::CMPSData, β::Real)
    Tψ = T*ψ
    var = real(ln_ovlp(Tψ, Tψ, β) + ln_ovlp(ψ, ψ, β) - 2 * ln_ovlp(ψ, Tψ, β))
    return var 
end

function boundary_cmps_var_optim(T::CMPO, init::CMPSData, β::Real; maxiter::Integer=1000, tol::Real=1e-9, verbosity::Integer=2, ϵ::Real=1e-6)

    function _f(ϕ::CMPSData)
        return variance(T, ϕ, β)
    end
    ψ1 = init + ϵ * CMPSData(rand, get_χ(init), get_d(init))
    
    # optimization 
    optalg = CircularCMPSRiemannian(maxiter, tol, verbosity)
    ψ1, var_result, grad, numfg, history = minimize(_f, ψ1, optalg)

    return ψ1, var_result, grad, numfg, history
end

function free_energy(T::CMPO, ψL::CMPSData, ψ::CMPSData, β::Real)
    f = real(ln_ovlp(ψL, T, ψ, β) - ln_ovlp(ψL, ψ, β)) / (-β)
    return f
end

function energy(T::CMPO, ψL::CMPSData, ψ::CMPSData, β::Real)
    Tψ = T*ψ
    K_leg3 = ψL * Tψ
    K_leg2 = ψL * ψ

    expK_leg3 = finite_env(K_leg3, β)[1]
    expK_leg2 = finite_env(K_leg2, β)[1]

    return -real(tr(expK_leg3 * K_leg3) - tr(expK_leg2 * K_leg2))
end

function klein(ϕL::CMPSData, ϕ::CMPSData, L::Real)
    Kmat = K_mat(ϕL, ϕ)
    _, α = finite_env(Kmat, L)
    ϕ = rescale(ϕ, -real(α), L)
    ϕL = rescale(ϕL, -real(α), L)

    # the Kmat for klein bottle entropy computation is different!
    IdL, IdR = id(space(ϕL)), id(space(ϕ)) 
    Kmat_klein = ϕL.Q ⊗ IdR + IdL ⊗ ϕ.Q + sum(ϕL.Rs .⊗ ϕ.Rs)
    expK_klein = exp(Kmat_klein * (L/2))
    @tensor gk = expK_klein[1 2; 2 1]
    sk = 2*real(log(gk))
    
    return sk
end

# only implemented for plain tensors
function direct_sum(A::MPSBondTensor, B::MPSBondTensor)
    χA, χB = dim(_firstspace(A)), dim(_firstspace(B))

    A_oplus_B = TensorMap(zeros, ComplexF64, ℂ^(χA+χB), ℂ^(χA+χB))
    A_oplus_B.data[1:χA, 1:χA] = A.data
    A_oplus_B.data[1+χA:end, 1+χA:end] = B.data
    return A_oplus_B
end

function direct_sum(ψ1::CMPSData, ψ2::CMPSData; α::Real=0)
    Id = id(domain(ψ2.Q))
    Q = direct_sum(ψ1.Q, ψ2.Q + α*Id)
    Rs = direct_sum.(ψ1.Rs, ψ2.Rs)
    return CMPSData(Q, Rs)
end
function direct_sum(ψ1::CMPSData, ψ2::CMPSData, Δ::Real, β::Real)
    α = log(Δ) / (2*β)
    return direct_sum(ψ1, ψ2; α = α)
end

function W_mul(W::Matrix{<:Number}, ψ::CMPSData)
    return CMPSData(ψ.Q, W * ψ.Rs)
end

function Base.:*(W::Matrix{<:Number}, ψ::CMPSData)
    return W_mul(W, ψ)
end
function Base.:*(W::Matrix{<:Number}, T::CMPO)
    Ls = W * T.Ls 
    Ps = W * T.Ps 
    return CMPO(T.Q, Ls, T.Rs, Ps) 
end
function Base.:*(T::CMPO, W::Matrix{<:Number})
    Rs = T.Rs * W 
    Ps = T.Ps * W 
    return CMPO(T.Q, T.Ls, Rs, Ps)
end

function TensorKit.inner(T1::CMPO, T2::CMPO, L::Real)
    Id_1, Id_2 = id(space(T1)), id(space(T2))

    Kmat = K_otimes(T1.Q, Id_2) + K_otimes(Id_1, T2.Q) + sum(K_otimes.(T1.Rs, T2.Rs)) + sum(K_otimes.(T1.Ls, T2.Ls))
    return finite_env(Kmat, L)[2] 
end