"""
    gauge_fixing_proj(ψ::cmps, L::Real; gauge::Symbol=:periodic) -> proj::TensorMap 

    Fix the gauge for a tangent vector tensor. `gauge` can be chosen between `:periodic` and `:left` .
    This function generates a projector `proj` that can be used to
    - parametrize a fixed-gauge tangent vector
    - eleminate the gauge degrees of freedom in an existing tangent vector
"""
function gauge_fixing_proj(ψ::cmps, L::Real; gauge::Symbol=:periodic)
    # tensor `A` is connected to impurity tensor in the tangent vector
    χ = get_chi(ψ)
    A = cmps(id(ℂ^χ), copy(ψ.R))
    A = convert_to_tensormap(A)

    if gauge == :periodic
        # density matrix ρ
        K = K_mat(ψ, ψ)
        ρ = finite_env(K, L)
        ρ = permute(ρ, (2, 3), (4, 1))
   
        # construct the projector
        @tensor Aρ[-1, ; -2, -3] := ρ[-1, 1, -2, 2] * A'[2, 1, -3]
        proj = rightnull(Aρ)'

    elseif gauge == :left
        # left dominant eigenvector 
        lopT = transf_mat_T(ψ, ψ)
        Vl = eigsolve(lopT, TensorMap(rand, ComplexF64, ℂ^χ, ℂ^χ), 1, :LR)[2][1]

        # projector
        @tensor Aρ[-1, ; -2, -3] := Vl'[1, -2] * A'[-1, 1, -3]
        proj = rightnull(Aρ)'
    end
    return proj
end

"""
    tangent_map(ψ::cmps, L::Real; gauge::Symbol=:periodic) -> Function

    tangent map. act on the parameter space obtained after gauge elimination. 
"""
function tangent_map(ψ::cmps, L::Real, p::Real=0; gauge::Symbol=:periodic)

    χ, d = get_chi(ψ), get_d(ψ)

    # diagonalize K matrix, normalize W according to length 
    K = K_mat(ψ, ψ)
    W, UR = eig(K)
    UL = inv(UR)

    # gauge fixing projector
    proj = gauge_fixing_proj(ψ, L; gauge=gauge)    

    # calculate coefficient matrix  
    Wvec = diag(W.data)
    normψ = logsumexp(L .* Wvec)
    Wvec .-= normψ / L
    @tullio coeffW[ix, iy] := theta2(L, Wvec[ix], Wvec[iy] - p*im)
    #coeffW = similar(W)
    #function coeff(a::Number, b::Number)
    #    # when using coeff.(avec, bvec'), avec is column vector, bvec is row vector
    #    # in this way the index ordering is consistent
    #    if a ≈ b'
    #        return L*exp(a*L)
    #    else 
    #        return (exp(a*L) - exp(b'*L)) / (a - b')
    #    end
    #end
    #copyto!(coeffW.data, coeff.(Wvec, (Wvec .- p*im)'))
    
    # A tensor that is connected to G
    A = cmps(id(ℂ^get_chi(ψ)), copy(ψ.R))
    A = convert_to_tensormap(A)

    D = id(ℂ^(d+1))
    D.data[1] = 0

    idW = id(ℂ^(χ^2))

    # calculate the map
    function f(V::AbstractTensorMap{T, S, 1, 1}) where {T, S}
        # crossing term
        @tensor M0[-1, -2; -3, -4] := V[1, -4] * proj[-2, 2, 1] * A'[-3, -1, 2]
        M = UL * M0 * UR
        #M = elem_mult(M, coeffW)
        #M = elem_mult_f1(M, (ix, iy) -> theta2(L, Wvec[ix], Wvec[iy] - p*im))
        @tullio M.data[ix, iy] = M.data[ix, iy] * coeffW[ix, iy]
        M = UR * M * UL
        M = permute(M, (2, 3), (4, 1))
        @tensor Vc[-1; -2] := M[1, 3, 2, -2] * A[2, 4, 1] * proj'[-1, 3, 4] 
        # diagonal term
        ρ = UR * exp(W*L - idW*normψ) * UL 
        ρ = permute(ρ, (2, 3), (4, 1))
        @tensor Vd[-1; -2] := ρ[3, 5, 2, -2] * proj[2, 4, 1] * V[1, 3] * D[6, 4] * proj'[-1, 5, 6]
        return L*(Vc + Vd) 
    end

    return f
end