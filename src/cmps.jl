abstract type AbstractCMPSData end

mutable struct CMPSData <: AbstractCMPSData 
    Q::MPSBondTensor
    Rs::Vector{<:MPSBondTensor}
end

"""
    cmps(f, chi::Integer, d::Integer) -> cmps

    generate a (plain) cmps with vitual dim `chi` and physical dim `d` using function `f` to generate the parameters.
    Example: `cmps(rand, 2, 4)` 
"""
function CMPSData(f, χ::Integer, d::Integer)
    Q = TensorMap(f, ComplexF64, ℂ^χ, ℂ^χ)
    Rs = MPSBondTensor{ComplexSpace}[]
    for ix in 1:d 
        push!(Rs, TensorMap(f, ComplexF64, ℂ^χ, ℂ^χ))
    end
    return CMPSData(Q, Rs)
end

# operations on the data. not on the cMPS
Base.:+(ψ::CMPSData, ϕ::CMPSData) = CMPSData(ψ.Q + ϕ.Q, ψ.Rs .+ ϕ.Rs)
Base.:+(ψ::CMPSData, ϕ::Base.RefValue) = (ψ + CMPSData(ϕ[].Q, ϕ[].Rs)) # TODO. used to fix autodiff in leading_boundary_cmps. fix _fg in optimalg
Base.:-(ψ::CMPSData, ϕ::CMPSData) = CMPSData(ψ.Q - ϕ.Q, ψ.Rs .- ϕ.Rs)
Base.:*(ψ::CMPSData, x::Number) = CMPSData(ψ.Q * x, ψ.Rs .* x)
Base.:*(x::Number, ψ::CMPSData) = CMPSData(ψ.Q * x, ψ.Rs .* x)
LinearAlgebra.dot(ψ1::CMPSData, ψ2::CMPSData) = dot(ψ1.Q, ψ2.Q) + sum(dot.(ψ1.Rs, ψ2.Rs))
LinearAlgebra.norm(ψ::CMPSData) = sqrt(norm(dot(ψ, ψ)))

function Base.similar(ψ::CMPSData) 
    Q = similar(ψ.Q)
    Rs = [similar(R) for R in ψ.Rs]
    return CMPSData(Q, Rs)
end

function randomize!(ψ::CMPSData)
    randomize!(ψ.Q)
    for R in ψ.Rs
        randomize!(R)
    end
end

function Base.zero(ψ::CMPSData) 
    Q = zero(ψ.Q)
    Rs = [zero(R) for R in ψ.Rs]
    return CMPSData(Q, Rs)
end

@inline get_χ(ψ::CMPSData) = dim(_firstspace(ψ.Q))
@inline get_d(ψ::CMPSData) = length(ψ.Rs)
TensorKit.space(ψ::CMPSData) = _firstspace(ψ.Q)

get_matrices(ψ::CMPSData) = (ψ.Q, ψ.Rs)

# TODO. define transfer_matrix object as MPSKit.jl did
""" 
    transfer_matrix(ϕ::CMPSData, ψ::CMPSData) where S

    The transfer matrix for <ϕ|ψ>.
    target at right vector. 
    T = I + ϵK. returns K
"""
function transfer_matrix(ϕ::CMPSData, ψ::CMPSData)
    function fK(v::MPSBondTensor)
        Tv = ψ.Q * v + v * ϕ.Q' 
        for (Rd, Ru) in zip(ψ.Rs, ϕ.Rs)
            Tv += Rd * v * Ru'
        end
        return Tv
    end
    return fK 
end

"""
   transfer_matrix_dagger(ϕ::CMPSData{S}, ψ::CMPSData{S}) where S
    
   The Hermitian conjugate of the transfer matrix for <ϕ|ψ>.
   target at left vector. 
   T† = I + ϵK†. returns K†
"""
function transfer_matrix_dagger(ϕ::CMPSData, ψ::CMPSData) 
    function fK_dagger(v::MPSBondTensor)
        Tv = v * ψ.Q + ϕ.Q' * v 
        for (Rd, Ru) in zip(ψ.Rs, ϕ.Rs)
            Tv += Ru' * v * Rd
        end
        return Tv
    end
    return fK_dagger
end

"""
    left_canonical(ψ::CMPSData)

    Convert the input cMPS into the left-canonical form. 
    Return the gauge transformation matrix X and the left-canonicalized cMPS. 
"""
function left_canonical(ψ::CMPSData)
    # transfer matrix dagger
    fK_dagger = transfer_matrix_dagger(ψ, ψ)
    
    # solve the fixed-point equation
    init = similar(ψ.Q, _firstspace(ψ.Q)←_firstspace(ψ.Q))
    randomize!(init);
    ws, vls, _ = eigsolve(fK_dagger, init, 1, :LR)
    w = ws[1]
    vl = vls[1]

    # obtain gauge transformation matrix X
    _, s, u = tsvd(vl)
    X = sqrt(s) * u
    Xinv = u' * sqrt(inv(s))

    # update Q and R 
    Q = X * ψ.Q * Xinv - 0.5 * w * id(_firstspace(ψ.Q)) # TODO. better way to normalize? 
    Rs = [X * R * Xinv for R in ψ.Rs]

    return X, CMPSData(Q, Rs) 
end

"""
    right_canonical(ψ::CMPSData) 

    Convert the input cmps into the right-canonical form. 
    Return the gauge transformation matrix Y and the right-canonicalized cmps. 
"""
function right_canonical(ψ::CMPSData)
    # transfer matrix
    fK = transfer_matrix(ψ, ψ)

    # solve the fixed point equation
    init = similar(ψ.Q, _firstspace(ψ.Q)←_firstspace(ψ.Q))
    randomize!(init);
    ws, vrs, _ = eigsolve(fK, init, 1, :LR)
    w = ws[1]
    vr = vrs[1]

    # obtain gauge transformation matrix Yinv
    _, s, u = tsvd(vr)
    Y = sqrt(s) * u
    Yinv = u' * sqrt(inv(s))

    # update Q and R 
    Q = Yinv' * ψ.Q * Y' - 0.5 * w * id(_firstspace(ψ.Q)) # TODO. better way to normalize? 
    Rs = [Yinv' * R * Y' for R in ψ.Rs]
    
    return Y, CMPSData(Q, Rs) 
end

"""
    expand(psi::cmps, chi::Integer, perturb::Float64=1e-3) -> cmps

    expand the cmps `psi` to a target bond dimension `chi` by adding small numbers of size `perturb`.
    Only works when the Q, Rs are plain tensors.
"""
function expand(ψ::CMPSData, χ::Integer, L::Real; perturb::Float64=1e-3)
    χ0, d = get_χ(ψ), get_d(ψ)
    if χ <= χ0
        @warn "new χ not bigger than χ0"
        return ψ
    end
    mask = similar(ψ.Q, ℂ^χ ← ℂ^χ)
    fill_data!(mask, randn)
    mask = perturb * mask
    Q = copy(mask)
    Q.data[1:χ0, 1:χ0] += ψ.Q.data
    for ix in χ0+1:χ
        Q.data[ix, ix] += 2*log(perturb)/L # suppress
    end

    Rs = MPSBondTensor[]
    for R0 in ψ.Rs
        R = similar(R0, ℂ^χ ← ℂ^χ)
        fill_data!(R, randn)
        R = perturb * R
        R.data[1:χ0, 1:χ0] += R0.data
        push!(Rs, R)
    end

    return CMPSData(Q, Rs) 
end

"""
    K_mat(ϕ::CMPSData, ψ::CMPSData)

    calculate the K_mat from two cmpses `phi` and `psi`. order of indices:

        -1 -->--   ϕ'  -->-- -3
                   |
                   ^
                   |
        -2 --<--   ψ   --<-- -4

    such tensor contraction is implemented in `K_otimes`  

    permute as `permute(K, (2, 3), (4, 1))` (implemented as `K_permute`) and get

        -4 -->--   ϕ'  -->-- -2
                   |
                   ^
                   |
        -1 --<--   ψ   --<-- -3
"""
function K_mat(ϕ::CMPSData, ψ::CMPSData)
    Id_ψ, Id_ϕ = id(space(ψ)), id(space(ϕ)) 
    Kmat = K_otimes(ϕ.Q, Id_ψ) + K_otimes(Id_ϕ, ψ.Q) + sum(K_otimes.(ϕ.Rs, ψ.Rs))
    return Kmat
end

"""
    finite_env(t::TensorMap{ComplexSpace}, L::Real)

    For a cMPS transfer matrix `t` (see `K_mat`), calculate the environment block for the finite length `L`.
    The input `t` matrix should look like 
    ``` 
        -1 -->--  phi' -->-- -3
                   |
                   ^
                   |
        -2 --<--  psi  --<-- -4
    ```
    This function will calculate exp(t) / tr(exp(t)) by diagonalizing `t`.
"""
function finite_env(K::TensorMap{ComplexSpace}, L::Real)
    W, UR = eig(K)
    UL = inv(UR)
    Ws = []

    if W.data isa Matrix 
        Ws = diag(W.data)
    elseif W.data isa TensorKit.SortedVectorDict
        Ws = vcat([diag(values) for (_, values) in W.data]...)
    end
    Wmax = maximum(real.(Ws))
    ln_of_norm = Wmax * L + log(sum(exp.(Ws .* L .- Wmax * L)))

    W = W - (ln_of_norm / L) * id(_firstspace(W)) 
    expW = exp(W * L)
   
    return UR * expW * UL, ln_of_norm
end

"""
    rescale(ψ::CMPSData, lnα::Real, L::Real)

    rescale the cMPS: ψ -> exp(lnα) * ψ
"""
function rescale(ψ::CMPSData, lnα::Real, L::Real)
    Q = ψ.Q + (lnα / 2 / L) * id(domain(ψ.Q))
    return CMPSData(Q, ψ.Rs)
end
