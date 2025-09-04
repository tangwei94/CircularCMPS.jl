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

for ix in 50:99
    @show ix, fidelity(ϕs[ix], ϕs[ix+2], β)
end

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

fidel1s, norm1s = Float64[], Float64[]
fidel2s, norm2s = Float64[], Float64[]
fidel3s, norm3s = Float64[], Float64[]

λs = zeros(ComplexF64, 10)
for ix0 in 1:88
    global λs  
    Ns = zeros(ComplexF64, 10, 10)
    ψs = ϕs[ix0+1:ix0+10]
    ψ0 = ϕs[ix0+11]

    Tψ = left_canonical(T*ψs[end])[2]
    ψ = left_canonical(ψs[end])[2]
    Tψ = direct_sum(Tψ, ψ; α=log(1)/β/2)
    _f(ψa::CMPSData) = fidelity(Tψ, ψa, β)
    function _fg(ϕa::CMPSData)
        ϕ = left_canonical(ϕa)[2]
        fvalue = _f(ϕ)
        ∂ϕ = _f'(ϕ)
        dQ = zero(∂ϕ.Q) 
        dRs = ∂ϕ.Rs .- ϕ.Rs .* Ref(∂ϕ.Q)

        return fvalue, CMPSData(dQ, dRs) 
    end
    if ix0 > 1
        ψnew = sum(ψs .* λs)
        push!(fidel1s, _f(ψnew))
        push!(fidel2s, _f(ψs[end]))
        push!(fidel3s, _f(ψ0))
        push!(norm1s, norm(_f'(ψnew)))
        push!(norm2s, norm(_f'(ψs[end])))
        push!(norm3s, norm(_f'(ψ0)))
    end

    for ix in 1:10, iy in 1:10
        Ns[ix, iy] = dot(ψs[ix], ψs[iy])
    end
    cs = dot.(ψs, Ref(ψ0))
    λs = quasi_inv(Ns, 1e-9) * cs

    ψ1 = sum(λs .* ψs)

    @show  _f(ψ1), norm(_f'(ψ1))
    @show ix0, λs
end
@save "tmp5.jld2" norm1s norm2s norm3s fidel1s fidel2s fidel3s
@load "tmp5.jld2" norm1s norm2s norm3s fidel1s fidel2s fidel3s

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 600))

ax1 = Axis(fig[1, 1], 
        xlabel = "power steps",
        ylabel = "grad norm",
        yscale = log10, 
        )
lines!(ax1, 1:length(norm1s), norm1s, label="new init")
lines!(ax1, 1:length(norm2s), norm2s, label="old init")
lines!(ax1, 1:length(norm3s), norm3s, label="optimized")

axislegend(ax1, position=:lb, framevisible=false)
@show fig

calc_err(x, x0) = abs(x-x0) / abs(x0)

ax2 = Axis(fig[2, 1], 
        xlabel = "power steps",
        ylabel = "init error",
        yscale = log10, 
        )
lines!(ax2, 1:length(norm1s), calc_err.(fidel1s, fidel3s), label="new init")
lines!(ax2, 1:length(norm2s), calc_err.(fidel2s, fidel3s), label="old init")

axislegend(ax2, position=:lb, framevisible=false)
@show fig