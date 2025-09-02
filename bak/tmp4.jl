# test diis


using LinearAlgebra, TensorKit
using OptimKit
using CairoMakie
using JLD2 
using Revise
using CircularCMPS

kept_states = 5

hz = 0.05
T, Wmat = xxz_af_cmpo(1; hz=hz)

α = 2^(1/4)
βs = 0.32 * α .^ (0:32)

β0 = βs[26]
@load "heisenberg/results/cooling/heisenberg_hz0.05_beta$(β0)-toles8.jld2" ψ
β = βs[27]

function quasi_inv(A::AbstractMatrix, ϵ::Float64)
    U, S, V = svd(A)
    s0 = S[1]
    Sinv = map(S) do s 
        if s < ϵ * s0
            return 0
        else
            return 1/s
        end 
    end
    return V * Diagonal(Sinv) * U'
end

function power_iteration1(T::CMPO, Wmat::Matrix{<:Number}, β::Real, ψ::CMPSData, alg::PowerMethod)
    printstyled("\n[ power_iteration: doing power method with 
        β = $(β)
        maxiter_power = $(alg.maxiter_power) 
        spect_shifting = $(alg.spect_shifting)
        maxχ = $(alg.maxχ)
        tol_fidel=$(alg.tol_fidel)
        tol_ES=$(alg.tol_ES)
        verbosity=$(alg.verbosity)
        maxiter_compress=$(alg.maxiter_compress) \n"; bold=true, color=:red)

    f, E, var, fidel = Inf, Inf, Inf, Inf
    ϕs = CMPSData[CircularCMPS.gauge_fixing(ψ, β)]
    λs = [zeros(ComplexF64, kept_states - 1); 1.0]

    for ix in 1:alg.maxiter_power
        # multiply T on ψ
        Tψ = left_canonical(T*ψ)[2]
        if alg.spect_shifting > 0
            ψ = left_canonical(ψ)[2]
            Tψ = direct_sum(Tψ, ψ; α=log(alg.spect_shifting)/β/2)
        end

        # decide bond dimension
        χ, err = CircularCMPS.suggest_χ(Tψ, β; tol=alg.tol_ES, maxχ=alg.maxχ, minχ=get_χ(ψ))
        printstyled("[ power_iteration: next χ: $(χ), possible error: $(err) \n"; bold=true)

        # initial guess for compression 
        if length(ϕs) == kept_states
            ψinit = sum(λs .* ϕs)
        else
            ψinit = ψ
        end

        # compression
        tol_compress = min(0.1*abs(fidel), 100*alg.tol_fidel) # gradually lowering the tolerance for compression to save time
        ψ1 = compress(Tψ, χ, β; init=ψinit, maxiter=alg.maxiter_compress, verbosity=alg.verbosity, tol=tol_compress)
        fidel = real(2*ln_ovlp(ψ, ψ1, β) - ln_ovlp(ψ, ψ, β) - ln_ovlp(ψ1, ψ1, β))

        # append new ψ 
        if dim(space(ϕs[end])) == dim(space(ψ1)) 
            ϕ1 = CircularCMPS.gauge_fixing(ϕs[end], ψ1, β)
            push!(ϕs, ϕ1)
        else
            ϕs = CMPSData[CircularCMPS.gauge_fixing(ψ1, β)]
        end

        @show length(ϕs)
        Ns = zeros(ComplexF64, kept_states, kept_states)
        if length(ϕs) == kept_states + 1
            for ix in 1:kept_states, iy in 1:kept_states
                Ns[ix, iy] = dot(ϕs[ix], ϕs[iy])
            end
            cs = dot.(ϕs[1:end-1], Ref(ϕs[end]))
            λs = quasi_inv(Ns, 1e-9) * cs
            deleteat!(ϕs, 1)
        end

        ψ = ψ1
        ψL = W_mul(Wmat, ψ)
        f = free_energy(T, ψL, ψ, β)
        E = energy(T, ψL, ψ, β)
        var = variance(T, ψ, β)
        printstyled("[ power_iteration: ix, f, E, var, fidel $(ix) $(f) $(E) $(var) $(fidel) \n"; color=:red)

        ix > 2 && abs(fidel) < alg.tol_fidel && break
    end
    return ϕs, ψ, f, E, var
end

ϕs, ψ1, _, _, _ =  power_iteration1(T, Wmat, β, ψ, PowerMethod(tol_ES=1e-8, tol_fidel=1e-8, maxiter_power=100))

function fidelity(ϕa::CMPSData, ϕb::CMPSData, L::Real)
    return real(-ln_ovlp(ϕa, ϕb, L) - ln_ovlp(ϕb, ϕa, L) + ln_ovlp(ϕa, ϕa, L) + ln_ovlp(ϕb, ϕb, L))
end

@save "tmp4.jld2" ϕs
@load "tmp4.jld2" ϕs

norm1s = Float64[]
norm2s = Float64[]
norm3s = Float64[]

for ix0 in 1:90
    Ns = zeros(ComplexF64, 5, 5)
    ψs = ϕs[ix0+1:ix0+5]
    ψ0 = ϕs[ix0+6]

    _f(ψa::CMPSData) = fidelity(T*ψs[end], ψa, β)
    if ix0 > 1
        ψnew = sum(ψs .* λs)
        @show _f(ψnew), norm(_f'(ψnew))
        @show _f(ψs[end]), norm(_f'(ψs[end]))
        @show _f(ψ0), norm(_f'(ψ0))
        push!(norm1s, norm(_f'(ψnew)))
        push!(norm2s, norm(_f'(ψs[end])))
        push!(norm3s, norm(_f'(ψ0)))
    end

    for ix in 1:5, iy in 1:5
        Ns[ix, iy] = dot(ψs[ix], ψs[iy])
    end
    cs = dot.(ψs, Ref(ψ0))
    λs = quasi_inv(Ns, 1e-9) * cs

    ψ1 = sum(λs .* ψs)

    @show  _f(ψ1), norm(_f'(ψ1))
    @show ix0, λs
end