using LinearAlgebra, TensorKit, KrylovKit 
using ChainRules, Zygote  
using CairoMakie 
using JLD2 
using OptimKit 
using Revise 
using CircularCMPS 

c1, μ1 = 1, 1. 
c2, μ2 = 1, 1. 
c12 = 0.5 

Hm = MultiBosonLiebLiniger([c1 c12; c12 c2], [μ1, μ2], Inf) 

χb = 2
χ = 4
jχ = 0

################# computation ####################

recorderD11 = ComplexF64[]
recorderD22 = ComplexF64[]
recorderD33 = ComplexF64[]
recorderD44 = ComplexF64[]
function myfinalize!(x, f, g, numiter)
    if numiter == 1
        empty!(recorderD11)
        empty!(recorderD22)
        empty!(recorderD33)
        empty!(recorderD44)
    end
    push!(recorderD11, x.data.Ds[1][1, 1])
    push!(recorderD22, x.data.Ds[1][2, 2])
    push!(recorderD33, x.data.Ds[1][3, 3])
    push!(recorderD44, x.data.Ds[1][4, 4])
end

ψ0 = MultiBosonCMPSData_MDMinv(rand, χ, 2);
ψ0 = left_canonical(ψ0);
ψ0.Ds[1]

resa = ground_state(Hm, ψ0; gradtol=1e-8, maxiter=400, _finalize! = myfinalize!, do_polar_retraction=true); 
resd = ground_state(Hm, ψ0; gradtol=1e-8, maxiter=400, _finalize! = myfinalize!, do_polar_retraction=false);
@show norm(resa[3])
@show norm(resd[3])

ψ0 = resa[1]
Q0 = ψ0.Q
R0s = Ref(ψ0.M) .* ψ0.Ds .* Ref(ψ0.Minv)
Q0 + Q0' + sum([R' * R for R in R0s])

ψ1 = expand(resa[1], χ+4; perturb = 1.0);
Q1 = ψ1.Q
R1s = Ref(ψ1.M) .* ψ1.Ds .* Ref(ψ1.Minv)
Q1 + Q1' + sum([R' * R for R in R1s]) |> norm
resa[2]

left_canonical(ψ1)
resd2 = ground_state(Hm, ψ1; gradtol=1e-8, maxiter=1000, _finalize! = myfinalize!, do_polar_retraction=false);

norm.(ψ1.Ds[1])
as = angle.(diag(ψ1.Ds[1]))
(as[1] + as[3]) / 2
norm.(ψ1.Ds[2])
as = angle.(diag(ψ1.Ds[2]))
(as[1] + as[3]) / 2

fig = Figure(size=(600, 600))
ax = Axis(fig[1, 1], 
        xlabel = "Re Dᵃᵢᵢ",
        ylabel = "Im Dᵃᵢᵢ",
        )
lines!(ax, real.(cos.(0:0.01:2*π)), sin.(0:0.01:2*π), color=:grey)
scatter!(ax, real.(recorderD11[1:end]), imag.(recorderD11[1:end]), label="D11")
scatter!(ax, real.(recorderD22[1:end]), imag.(recorderD22[1:end]), label="D22")
scatter!(ax, real.(recorderD33[1:end]), imag.(recorderD33[1:end]), label="D33")
scatter!(ax, real.(recorderD44[1:end]), imag.(recorderD44[1:end]), label="D44")
scatter!(ax, real(recorderD11[end]), imag(recorderD11[end]), marker=:star5, color=:red)
scatter!(ax, real(recorderD22[end]), imag(recorderD22[end]), marker=:star5, color=:red)
scatter!(ax, real(recorderD33[end]), imag(recorderD33[end]), marker=:star5, color=:red)
scatter!(ax, real(recorderD44[end]), imag(recorderD44[end]), marker=:star5, color=:red)
#axislegend(ax, position=:rt)
@show fig

recorderD11[end] |> norm
recorderD22[end] |> norm
recorderD33[end] |> norm
recorderD44[end] |> norm

perm = sortperm(diag(norm.(resa[1].Ds[1])))
a = diag(resa[1].Ds[1])[perm]
@show norm.(a[1:2:end])
@show angle(a[2]/a[1]) 
@show angle(a[4]/a[3]) 

fig = Figure(backgroundcolor = :white, fontsize=14, resolution= (800, 400))
ax1 = Axis(fig[1, 1], 
        xlabel = "Re Dᵃᵢᵢ",
        ylabel = "Im Dᵃᵢᵢ",
        )
ax2 = Axis(fig[1, 2], 
        xlabel = "Re Dᵇᵢᵢ",
        ylabel = "Im Dᵇᵢᵢ",
        )

ψ2 = resa[1]
norm.(diag(ψ2.Ds[1]))
θs = 0:0.01:2*π
for ix in 1:2
    r = norm.(diag(ψ2.Ds[1]))[2*ix]
    @show r 
    lines!(ax1, r*cos.(θs), r*sin.(θs))
end
scatter!(ax1, real.(diag(ψ2.Ds[1])), imag.(diag(ψ2.Ds[1])), color=:black, label="Dᵃ")

for ix in 1:2
    r = norm.(diag(ψ2.Ds[2]))[2*ix]
    @show r 
    lines!(ax2, r*cos.(θs), r*sin.(θs))
end
scatter!(ax2, real.(diag(ψ2.Ds[2])), imag.(diag(ψ2.Ds[2])), color=:black, label="Dᵃ")

@show fig

eigvals(ψ2.Minv * ψ2.Q * ψ2.M)

ψ2 = MultiBosonCMPSData_MDMinv(rand, 6, 2);
ψ2 = left_canonical(ψ2);
ψ2 = resc[1];
resc = ground_state(Hm, ψ2; gradtol=1e-8, maxiter=500);
ψ3 = resc[1];

norm.(diag(resc[1].Ds[1]))
norm.(diag(resc[1].Ds[2]))




ψ2 = MultiBosonCMPSData_P(rand, χb, 2);
res = ground_state(Hm, ψ2; gradtol=1e-8, maxiter=1000); 
#res_wop = ground_state(Hm, ψ1; do_preconditioning = false, gradtol=1e-8, maxiter=250, m_LBFGS=25); 
@save "multiboson/results/MDMinv_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res
@save "multiboson/results/MDMinv_wop_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res_wop

χ = 8
ψ1 = MultiBosonCMPSData_MDMinv(rand, χ, 2);
res = ground_state(Hm, ψ1; gradtol=1e-8, maxiter=250); 

χ = 16
@load "multiboson/results/MDMinv_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_8.jld2" res
ψ3 = expand(res[1], χ, perturb = 1e-4);
ϕ3 = left_canonical(CMPSData(ψ3))[2];
res_lm = ground_state(Hm, ϕ3; Λs=sqrt(10) .^ (4:10), gradtol=1e-2, do_benchmark=true);
@save "multiboson/results/lagrangian_multiplier_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res_lm

ψ3 = MultiBosonCMPSData_MDMinv(res_lm[1]);
res = ground_state(Hm, ψ3; gradtol=1e-8, maxiter=1000); 
res_wop = ground_state(Hm, ψ3; do_preconditioning=false, gradtol=1e-8, maxiter=1000); 
@save "multiboson/results/MDMinv_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res
@save "multiboson/results/MDMinv_wop_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res_wop

################# analysis 1: benchmark MDMinv ####################
χ = 16
@load "multiboson/results/MDMinv_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res
@load "multiboson/results/MDMinv_wop_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res_wop
#@load "multiboson/results/lagrange_multiplier_further_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res_lagrange
Es, gnorms = res[5][:, 1], res[5][:, 2]
Es_wop, gnorms_wop = res_wop[5][:, 1], res_wop[5][:, 2]
#Es_lagrange, gnorms_lagrange = res_lagrange[5][:, 1], res_lagrange[5][:, 2]

fig = Figure(backgroundcolor = :white, fontsize=14, resolution= (600, 600))

gf = fig[1:5, 1] = GridLayout()
gl = fig[6, 1] = GridLayout()

ax1 = Axis(gf[1, 1], 
        xlabel = "steps",
        ylabel = "energy",
        )
lines!(ax1, 1:length(Es), Es, label="w/ precond.")
lines!(ax1, 1:length(Es_wop), Es_wop, label="w/o  precond.")
#lines!(ax1, 1:length(Es_lagrange), Es_lagrange, label="further increasing Λ")
@show fig

ax2 = Axis(gf[2, 1], 
        xlabel = "steps",
        ylabel = "gnorm",
        yscale = log10,
        )
lines!(ax2, 1:length(gnorms), gnorms, label="w/ precond.")
lines!(ax2, 1:length(gnorms_wop), gnorms_wop, label="w/o precond.")
#lines!(ax2, 1:length(gnorms_lagrange), gnorms_lagrange, label="further increasing Λ")
#axislegend(ax2, position=:rb)
@show fig

Legend(gl[1, 1], ax1, nbanks=2)
@show fig
save("multiboson/results/result_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).pdf", fig)

####### analysis 2: benchmark the Lagrange multiplier step ############
χ = 8
@load "multiboson/results/lagrangian_multiplier_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res_lm
@load "multiboson/results/lagrangian_multiplier_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ)-1.jld2" res_lm1
@load "multiboson/results/lagrangian_multiplier_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ)-2.jld2" res_lm2
Es, gnorms = res_lm[5][:, 1], res_lm[5][:, 2]
Es1, gnorms1 = res_lm1[5][:, 1], res_lm1[5][:, 2]
Es2, gnorms2 = res_lm2[5][:, 1], res_lm2[5][:, 2]

fig = Figure(backgroundcolor = :white, fontsize=14, resolution= (400, 600))

gf = fig[1:5, 1] = GridLayout()
gl = fig[6, 1] = GridLayout()

ax1 = Axis(gf[1, 1], 
        xlabel = "steps",
        ylabel = "energy",
        )
ylims!(ax1, -2.13, -2.00)
lines!(ax1, 1:length(Es), Es, label="Λ=1e2->1e5, tol=1e-2")
lines!(ax1, 1:length(Es1), Es1, label="Λ=1e5, tol=1e-2")
lines!(ax1, 1:length(Es2), Es2, label="Λ=1e2->1e5, tol=1e-4")
@show fig

ax2 = Axis(gf[2, 1], 
        xlabel = "steps",
        ylabel = "gnorm",
        yscale = log10,
        )
lines!(ax2, 1:length(gnorms),  gnorms, label="Λ=1e2->1e5, tol=1e-2")
lines!(ax2, 1:length(gnorms1), gnorms1, label="Λ=1e5, tol=1e-2")
lines!(ax2, 1:length(gnorms2), gnorms2, label="Λ=1e2->1e5, tol=1e-4")
#axislegend(ax2, position=:rb)
@show fig

Legend(gl[1, 1], ax1, nbanks=2)
@show fig
save("multiboson/results/result_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).pdf", fig)



function polar_retraction(a::Number, b::Number, α::Real)
    jac = [real(a) -imag(a) ; imag(a) real(a)]
    κ, ϕ = jac \ [real(b); imag(b)]
    return a * exp(α * (κ + im * ϕ))
end