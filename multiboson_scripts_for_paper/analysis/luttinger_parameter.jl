using LinearAlgebra, TensorKit, KrylovKit 
using ChainRules, Zygote 
using CairoMakie 
using JLD2 
using OptimKit 
using CSV, DataFrames
using Polynomials
using Revise 
using CircularCMPS 

"""
    second_derivative_nonuniform(xnodes, fnodes, x0)

Approximate f''(x0) given 5 data points (xnodes[i], fnodes[i])
that may be non-uniformly spaced.

Arguments:
- `xnodes`: length-5 vector of x positions
- `fnodes`: length-5 vector of f(x) values
- `x0`: point where you want the second derivative
"""
function second_derivative_nonuniform(xnodes, fnodes, x0)
    @assert length(xnodes) == 5 && length(fnodes) == 5 "Need exactly 5 nodes"
    h = xnodes .- x0
    # Build Vandermonde matrix: A[m+1,j] = h_j^m, m=0:4
    A = [h[j]^m for m in 0:4, j in 1:5]
    b = [0.0, 0.0, 2.0, 0.0, 0.0]      # second derivative of monomials at 0
    w = A \ b                          # finite difference weights
    return dot(w, fnodes)              # f''(x0) ≈ Σ w_j f_j
end


c = 1.0
μ, c12 = 1.0, 0.3
root_folder = "data_two_component_lieb_liniger" 

folders = Dict(
    "00" => "results_c$(c)_mu$(μ)_coupling$(c12)", 
    "pp" => "perturbing_pp_1e-2_results_c$(c)_mu$(μ)_coupling$(c12)", 
    "0p" => "perturbing_0p_1e-2_results_c$(c)_mu$(μ)_coupling$(c12)",
    "m0" => "perturbing_m0_1e-2_results_c$(c)_mu$(μ)_coupling$(c12)",
    "mm" => "perturbing_mm_1e-2_results_c$(c)_mu$(μ)_coupling$(c12)", 
    "mp" => "perturbing_mp_1e-2_results_c$(c)_mu$(μ)_coupling$(c12)"
)

c1, c2 = c, c
μ1, μ2 = μ, μ 

Es = Dict{String, Float64}()
n1s = Dict{String, Float64}()
n2s = Dict{String, Float64}()

for label in ["00", "pp", "0p", "m0", "mp", "mm"]
    folder = folders[label]
    df = CSV.read("$(joinpath(root_folder, folder))/basic_measurements.txt", DataFrame)
    df = rename!(df, Symbol.(strip.(string.(names(df)))))
    if df.chi[end] isa String
        E = parse(Float64, df.energy[end])
        n1 = parse(Float64, df.n1[end])
        n2 = parse(Float64, df.n2[end])
    else
        E = df.energy[end]
        n1 = df.n1[end]
        n2 = df.n2[end]
    end
    Es[label] = E
    n1s[label] = n1
    n2s[label] = n2
end

for label in ["0p", "m0", "mp"]
    label1 = "$(label[2])$(label[1])"

    Es[label1] = Es[label]
    n1s[label1] = n2s[label]
    n2s[label1] = n1s[label]
    
end

# K+
get_Xp(label) = n1s[label] + n2s[label]
get_Xm(label) = n1s[label] - n2s[label]
get_Y(label) = Es[label]
labels_Kplus = ["mm", "m0", "00", "0p", "pp"]
Xps = get_X.(labels_Kplus)
Yps = get_Y.(labels_Kplus)
sorted_indices = sortperm(Xps)
Xps_sorted = Xps[sorted_indices]
Yps_sorted = Yps[sorted_indices]
second_derivative = second_derivative_nonuniform(Xps_sorted, Yps_sorted, Xps_sorted[3])
println("Second derivative at central point: $(second_derivative)")

# K- 
get_X(label) = n1s[label] - n2s[label]
get_Y(label) = Es[label]
labels_Kplus = ["mp", "0p", "00", "p0", "pm"]
Xps = get_X.(labels_Kplus)
Yps = get_Y.(labels_Kplus)

fig = Figure()
scatter(Xps, Yps)

sorted_indices = sortperm(Xps)
Xps_sorted = Xps[sorted_indices]
Yps_sorted = Yps[sorted_indices]
second_derivative = second_derivative_nonuniform(Xps_sorted, Yps_sorted, Xps_sorted[3])
println("Second derivative at central point: $(second_derivative)")


# Alternative: use polynomial coefficients for exact second derivative
p = fit(Xps_sorted, Yps_sorted, 5)
X0 = Xps_sorted[3]
println("Second derivative from polynomial: $(2*p[2] + 6*p[3]*X0^1 + 12*p[4]*X0^2 + 20*p[5]*X0^3)")



