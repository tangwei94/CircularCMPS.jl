using LinearAlgebra, TensorKit
using CairoMakie
using JLD2 
using Revise
using CircularCMPS

hz = 1.5
#hz = parse(Float64, ARGS[1])
T, Wmat = xxz_fm_cmpo(1; hz=hz)

α = 2^(1/4)
βs = 0.32 * α .^ (0:32)

βs1 = βs[1:end] 

function calc_entanglement(index)
    SLRs, SRRs = Float64[], Float64[]
    for β in βs1
        @load "heisenberg_FMXY/double_sided_chi/data/heisenberg_hz$(hz)_beta$(β)-$(index).jld2" β f E var ψ

        ψL = Wmat * ψ
        ΛLR = half_chain_singular_values(ψL, ψ, β)
        ΛRR = half_chain_singular_values(ψ, β)

        SLR = - real(tr(ΛLR * log(ΛLR)))
        SRR = - real(tr(ΛRR * log(ΛRR)))
        push!(SLRs, SLR)
        push!(SRRs, SRR)
    end
    return SLRs, SRRs
end

function get_χs(index)
    χs = Int[]
    for β in βs1
        @load "heisenberg_FMXY/double_sided_chi/data/heisenberg_hz$(hz)_beta$(β)-$(index).jld2" ψ

        push!(χs, dim(space(ψ.Q, 1)))
    end
    return χs
end
function get_vars(index)
    vars = Float64[]
    for β in βs1
        @load "heisenberg_FMXY/double_sided_chi/data/heisenberg_hz$(hz)_beta$(β)-$(index).jld2" var

        push!(vars, var)
    end
    return vars
end

SLRs_8, SRRs_8 = calc_entanglement("toles9-spect_shifting1");
#SLRs_9, SRRs_9 = calc_entanglement("toles9");

get_χs("toles9-spect_shifting1")
get_vars("toles9-spect_shifting1")

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (400, 400))

ax1 = Axis(fig[1, 1], 
        xlabel = L"β",
        ylabel = L"S", 
        xscale = log10
        )

#lines!(ax1, βs1, SLRs_8, label=L"\text{SLR, tolES8}")
lines!(ax1, βs1, SRRs_8, label=L"\text{SRR, tolES8}")
axislegend(ax1, position=:rt, framevisible=false)
@show fig

save("heisenberg_FMXY/double_sided_chi/heisenberg_hz$(hz)_result.pdf", fig)

# ---> 
# when h_z >= 2.0, the system is polarized in the ground state, and there is a gap in the energy spectrum. SLR goes to zero at large beta, and there is a peak at finite temperature.
# however, it appears SRR still has a finite value at large beta, showing different entanglement structure 
# when h_z < 2.0, the ground state is not polarized. the system has gapless spin excitations
# <--- 

function show_analysis_results(ψL, ψ, β)

    χ2 = length(ψ.Q.data)

    scattering, reweighted = CircularCMPS.half_chain_singular_values_testtool(ψL, ψ, β);

    scattering_norms = norm.(scattering.data)
    scattering_norms /= maximum(scattering_norms)
    reweighted_norms = norm.(reweighted.data)
    reweighted_norms /= maximum(reweighted_norms)
    fig1, ax, hm = heatmap(log10.(scattering_norms), colorrange=(-3, 0), colormap=:Blues)
    Colorbar(fig1[:, end+1], hm)
    fig2, ax, hm = heatmap(log10.(reweighted_norms), colorrange=(-6, 0), colormap=:Blues)
    Colorbar(fig2[:, end+1], hm)

    return fig1, fig2
end

### M matrix
β = βs[end]
@load "heisenberg_FMXY/double_sided_chi/data/heisenberg_hz$(hz)_beta$(β)-toles9-spect_shifting1.jld2" ψ
ψL = Wmat * ψ

fig, fig2 = show_analysis_results(ψL, ψ, β);
@show fig
@show fig2

save("heisenberg_FMXY/double_sided_chi/heisenberg_hz$(hz)_MA.pdf", fig)

### plot entanglement spectrum
ΛLR = half_chain_singular_values(ψL, ψ, β)
ΛRR = half_chain_singular_values(ψ, β)
SLR = - real(tr(ΛLR * log(ΛLR)))
SRR = - real(tr(ΛRR * log(ΛRR)))
λLR = reverse(real.(diag(ΛLR.data)))
λRR = reverse(real.(diag(ΛRR.data)))

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (400, 400))
ax1 = Axis(fig[1, 1], 
        xlabel = L"i",
        ylabel = L"\lambda_i", 
        yscale = log10,
        )
scatter!(ax1, 1:length(λLR), λLR, marker='o', label=L"\text{LR}")
scatter!(ax1, 1:length(λRR), λRR, marker='x', label=L"\text{RR}")
axislegend(ax1, position=:rt, framevisible=false)
@show fig

λRR
λLR