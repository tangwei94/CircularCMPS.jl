using LinearAlgebra, TensorKit
using OptimKit
using CairoMakie
using JLD2 
using Revise
using CircularCMPS

kept_states = 10

hz = 0.05
T, Wmat = xxz_af_cmpo(1; hz=hz)

α = 2^(1/4)
βs = 0.32 * α .^ (0:32)

β0 = βs[33]
β = βs[33]

function fidelity(ϕa::CMPSData, ϕb::CMPSData, L::Real)
    return real(-ln_ovlp(ϕa, ϕb, L) - ln_ovlp(ϕb, ϕa, L) + ln_ovlp(ϕa, ϕa, L) + ln_ovlp(ϕb, ϕb, L))
end

@load "tmp4.jld2" ϕs

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

λs = zeros(ComplexF64, 5)
Ns = zeros(ComplexF64, 5, 5)
ψs = ϕs[91:95]
ψ0 = ϕs[96]

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 400))

ax1 = Axis(fig[1, 1], 
        xlabel = "power steps",
        ylabel = "element",
        )
lines!(ax1, 91:95, [real(ϕ.Rs[3][7]) for ϕ in ϕs[91:95]])
@show fig


for ix in 1:5, iy in 1:5
    Ns[ix, iy] = dot(ψs[ix], ψs[iy])
end
cs = dot.(ψs, Ref(ψ0))
λs = quasi_inv(Ns, 1e-9) * cs

norm(ψ0 - sum(λs .* ψs))/ norm(ψ0)
fidelity(ψ0, sum(λs .* ψs), β)


ψs = ϕs[92:96]
Tψ = left_canonical(T*ψs[end])[2]
ψ = left_canonical(ψs[end])[2]
Tψ = direct_sum(Tψ, ψ; α=log(1)/β/2)

fidel = fidelity(ψs[end], ψs[end-1], β)

ψ1 = compress(Tψ, 17, β; init=ψs[end], maxiter=200, verbosity=2, tol=0.1*fidel);
ψ2 = compress(Tψ, 17, β; init=sum(λs .* ψs), maxiter=200, verbosity=2, tol=0.1*fidel);
