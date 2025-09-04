c1, μ1 = 1., 2.
c2, μ2 = 1.5, 2.5
c12 = 0.5

@load "multiboson/results/preconditioned_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_12.jld2" res_wp
@load "multiboson/results/unpreconditioned_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_12.jld2" res_wop

Es_wp, gnorms_wp = res_wp[5][:, 1], res_wp[5][:, 2]
Es_wop, gnorms_wop = res_wop[5][:, 1], res_wop[5][:, 2]

fig = Figure(backgroundcolor = :white, fontsize=14, resolution= (400, 600))

gf = fig[1:5, 1] = GridLayout()
gl = fig[6, 1] = GridLayout()

ax1 = Axis(gf[1, 1], 
        xlabel = "steps",
        ylabel = "energy",
        )
lin1 = lines!(ax1, 1:length(Es_wp), Es_wp, label="w/ precond.")
lin2 = lines!(ax1, 1:length(Es_wop), Es_wop, label="w/o  precond.")
#axislegend(ax1, position=:rt)
@show fig

ax2 = Axis(gf[2, 1], 
        xlabel = "steps",
        ylabel = "gnorm",
        yscale = log10,
        )
lines!(ax2, 1:length(gnorms_wp), gnorms_wp, label="w/ precond.")
lines!(ax2, 1:length(gnorms_wop), gnorms_wop, label="w/o precond.")
#axislegend(ax2, position=:rt)
@show fig

Legend(gl[1, 1], ax1, nbanks=2)
@show fig
save("multiboson/results/result_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_12.pdf", fig)

let H=Hm

    cs = Matrix{ComplexF64}(H.cs)
    μs = Vector{ComplexF64}(H.μs)

    function fE_inf(ψ::MultiBosonCMPSData_MDMinv)
        ψn = CMPSData(ψ)
        OH = kinetic(ψn) + H.cs[1,1]* point_interaction(ψn, 1) + H.cs[2,2]* point_interaction(ψn, 2) + H.cs[1,2] * point_interaction(ψn, 1, 2) + H.cs[2,1] * point_interaction(ψn, 2, 1) - H.μs[1] * particle_density(ψn, 1) - H.μs[2] * particle_density(ψn, 2)
        TM = TransferMatrix(ψn, ψn)
        envL = permute(left_env(TM), (), (1, 2))
        envR = permute(right_env(TM), (2, 1), ()) 
        return real(tr(envL * OH * envR) / tr(envL * envR))
    end
    
    function fgE(ψ::MultiBosonCMPSData_MDMinv)
        E, ∂ψ = withgradient(fE_inf, ψ)
        @show norm(∂ψ[1])
        g = CircularCMPS.diff_to_grad(ψ, ∂ψ[1])
        return E, g
    end

    @show get_χ(res_wp[1])
    E, g = fgE(res_wp[1])
    println("E=$(E), gnorm=$(norm(g))")
    ϕ = expand(res_wp[1], 16; perturb=1e-3)
    E, g = fgE(ϕ)
    println("E=$(E), gnorm=$(norm(g))")
end