abstract type AbstractHamiltonian end

struct SingleBosonLiebLiniger <: AbstractHamiltonian
    c::Real
    μ::Real
    L::Real
end

function ground_state(H::SingleBosonLiebLiniger, ψ0::CMPSData)
    if H.L == Inf 
        function fE_inf(ψ::CMPSData)
            OH = kinetic(ψ) + H.c*point_interaction(ψ) - H.μ * particle_density(ψ)
            TM = TransferMatrix(ψ, ψ)
            envL = permute(left_env(TM), (), (1, 2))
            envR = permute(right_env(TM), (2, 1), ()) 
            return real(tr(envL * OH * envR) / tr(envL * envR))
        end
        @show "infinite system"

        return minimize(fE_inf, ψ0, CircularCMPSRiemannian(1000, 1e-9, 2)) # TODO. change this as input. 
    else
        @show "finite system of size $(H.L)"
        function fE_finiteL(ψ::CMPSData)
            OH = kinetic(ψ) + H.c*point_interaction(ψ) - H.μ * particle_density(ψ)
            expK, _ = finite_env(K_mat(ψ, ψ), H.L)
            return real(tr(expK * OH))
        end 

        return minimize(fE_finiteL, ψ0, CircularCMPSRiemannian(1000, 1e-9, 2)) # TODO. change this as input. 
    end
end

struct MultiBosonLiebLiniger <: AbstractHamiltonian
    cs::Matrix{<:Real}
    μs::Vector{<:Real}
    L::Real
end

