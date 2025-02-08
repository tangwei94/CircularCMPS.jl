using LinearAlgebra, TensorKit, KrylovKit
using CairoMakie
using JLD2 
using Revise
using CircularCMPS

Γ = 1.0
T, Wmat = ising_cmpo(Γ)

β = 28.963093757401015
χs = [10, 12, 16]

for χ in χs
    @load "ising/results/ising_Gamma$(Γ)_beta$(β)-chi$(χ).jld2" β f E var ψ
    @show β, f, E, var
    ψ, f, E, var = leading_boundary(T, β, ψ, VariationalOptim(verbosity=2))
    @save "ising/results/variational_ising_Gamma$(Γ)_beta$(β)-chi$(χ).jld2" β f E var ψ
    @show β, f, E, var
end

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 800))

ax1 = Axis(fig[1, 1], 
        xlabel = L"χ",
        ylabel = L"βΔE", 
        )
ΔEs = map(χs) do χ
    @load "ising/results/ising_Gamma$(Γ)_beta$(β)-chi$(χ).jld2" ψ
    Es = eigvals(convert(Array, K_mat(ψ, ψ)))
    abs(Es[end-1] - Es[end])
end
scatterlines!(ax1, χs, β .* ΔEs, label="β=$β")
lines!(ax1, χs, χs .* 0 .+ sqrt(2))
axislegend(ax1, position=:rt)
@show fig

βs = [32, 48, 64]
χs = 4:2:20

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 800))
ax1 = Axis(fig[1, 1], 
        xlabel = L"χ",
        ylabel = L"βΔE", 
        )
for β in βs
    ΔEs = map(χs) do χ
        @load "ising/results/ising_Gamma$(Γ)_beta$(β)-chi$(χ).jld2" ψ
        Es = eigvals(convert(Array, K_mat(ψ, ψ)))
        a = abs(Es[end-1] - Es[end])
        b = abs(Es[end-2] - Es[end])
        @show b, a, b/a
        return b/a
    end
    scatterlines!(ax1, χs, ΔEs, label="β=$β")
end
lines!(ax1, χs, χs .* 0 .+ (4+sqrt(2))/sqrt(2))
axislegend(ax1, position=:rt)
@show fig

β = 64
χs = 10:2:20

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 800))
ax1 = Axis(fig[1, 1], 
        xlabel = L"χ",
        ylabel = L"βΔE", 
        )
for ix in 1:5
    ΔEs = map(χs) do χ
        @load "ising/results/ising_Gamma$(Γ)_beta$(β)-chi$(χ).jld2" ψ 
        Es = eigvals(convert(Array, K_mat(ψ, ψ)))
        a = abs(Es[end-ix] - Es[end])
        return a
    end
    scatterlines!(ax1, χs, ΔEs .* β, label="β=$β")
end
lines!(ax1, χs, χs .* 0 .+ sqrt(2))
lines!(ax1, χs, χs .* 0 .+ (4+sqrt(2)))
axislegend(ax1, position=:rt)
@show fig