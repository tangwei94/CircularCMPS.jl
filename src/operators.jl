"""
    kinetic(ψ::CMPSData)

    Construct the tensor for kinetic energy density `(dψ† / dx)(dψ / dx)`.
"""
function kinetic(ψ::CMPSData)
    Q, Rs = get_matrices(ψ) 
    Oψ = Ref(Q) .* Rs - Rs .* Ref(Q)
    return sum(K_otimes.(Oψ, Oψ))
end

"""
    particle_density(ψ::CMPSData)

    Construct the tensor for particle density `ψ†ψ`.
"""
function particle_density(ψ::CMPSData)
    _, Rs = get_matrices(ψ) 
    return sum(K_otimes.(Rs, Rs))
end
function particle_density(ψ::CMPSData, μs::Vector)
    _, Rs = get_matrices(ψ) 
    return sum(K_otimes.(Rs, Rs) .* μs)
end
function particle_density(ψ::CMPSData, index::Integer)
    _, Rs = get_matrices(ψ) 
    return K_otimes(Rs[index], Rs[index])
end

"""
    point_interaction(ψ::CMPSData) 

    Construct the tensor for point interaction potential `ψ†ψ†ψψ`. 
"""
function point_interaction(ψ::CMPSData)
    _, Rs = get_matrices(ψ) 
    Oψ = Rs .* Rs
    return sum(K_otimes.(Oψ, Oψ))
end
# error
#function point_interaction(ψ::CMPSData, cs::Matrix)
    #_, Rs = get_matrices(ψ) 
    #R2s = Rs .* Rs 
    #return sum(K_otimes.(R2s, cs * R2s))
#end
function point_interaction(ψ::CMPSData, index::Integer)
    _, Rs = get_matrices(ψ) 
    Oψ = Rs[index] * Rs[index]
    return K_otimes(Oψ, Oψ)
end
function point_interaction(ψ::CMPSData, index1::Integer, index2::Integer)
    if index1 == index2
        return point_interaction(ψ, index1)
    end

    _, Rs = get_matrices(ψ)
    Oψ_1 = Rs[index1] * Rs[index2]
    Oψ_2 = Rs[index1] * Rs[index2]
    return K_otimes(Oψ_1, Oψ_2) 
end

"""
    pairing(ψ::CMPSData)

    Construct the tensor for pairint term `ψ†ψ†+ψψ`. 
"""
function pairing(ψ::CMPSData)
    Iψ = id(domain(ψ.Q))
    _, Rs = get_matrices(ψ) 
    Oψ = Rs .* Rs
    return sum(K_otimes.(Ref(Iψ), Oψ)) + sum(K_otimes.(Oψ, Ref(Iψ)))
end
function pairing(ψ::CMPSData, index::Integer)
    Iψ = id(domain(ψ.Q))
    _, Rs = get_matrices(ψ) 
    Oψ = Rs[index] * Rs[index]
    return K_otimes(Iψ, Oψ) + K_otimes(Oψ, Iψ)
end

"""
    penalty_term(ψ::CMPSData, index1::Integer, index2::Integer, Λ::Real)

    Lagrangian multiplier term `Λ [ψ1, ψ2]† [ψ1, ψ2]`.
"""
function penalty_term(ψ::CMPSData, index1::Integer, index2::Integer, Λ::Real; order::Integer=1)
    _, Rs = get_matrices(ψ) 
    Oψ = Rs[index1] * Rs[index2] - Rs[index2] * Rs[index1]
    O_penalty1 = K_otimes(Oψ, Oψ) * Λ
    O_penalty = deepcopy(O_penalty1)
    for _ in 2:order
        O_penalty = O_penalty1 * O_penalty + O_penalty1
    end
    return O_penalty
end

function penalty_term_type2(ψ::CMPSData, index1::Integer, index2::Integer, Λ::Real)
    _, Rs = get_matrices(ψ) 
    Oψ = Rs[index1] * Rs[index2] - Rs[index2] * Rs[index1]
    O_penalty = K_otimes(Oψ, Oψ)
    Id = id(domain(O_penalty))
    @show space(exp(Λ * O_penalty) - Id)
    return exp(Λ * O_penalty) - Id
end
