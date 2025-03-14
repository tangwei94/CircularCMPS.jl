using JLD2
using CairoMakie

function load_num_steps_for(χ::Integer)
    num_steps = Int[]
    for ix in 1:20
        @load "multiboson_scripts_for_paper/benchmark/data/benchmark_plain_opt_$(χ)_$(ix).jld2" res
        push!(num_steps, length(res[1]))
    end
    return num_steps
end

averaged_num_steps = []
for χ in [4, 6, 8]
    num_steps = load_num_steps_for(χ)
    avg_steps = sum(num_steps) / length(num_steps)
    push!(averaged_num_steps, avg_steps)
    println("χ = $χ: Average steps = $(round(avg_steps, digits=2)) ")
end

# Create a simple plot of χ vs averaged_num_steps
fig_simple = Figure(resolution=(800, 600))
ax_simple = Axis(fig_simple[1, 1], 
    xlabel="log(χ)", 
    ylabel="log(Average number of steps)",
    title="Optimization steps vs bond dimension")

χ_values = [4, 6, 8]
scatter!(ax_simple, log.(χ_values), log.(averaged_num_steps), markersize=15, color=:blue)
lines!(ax_simple, log.(χ_values), log.(averaged_num_steps), linewidth=2, color=:blue)

@show fig_simple
save("multiboson_scripts_for_paper/benchmark/data/steps_vs_chi.pdf", fig_simple)




