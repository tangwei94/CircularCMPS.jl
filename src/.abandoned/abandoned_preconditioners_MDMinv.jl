# failed preconditioners for multibosonCMPSData_MDMinv 
# copied from 


    # preconditioner type 3, 4, 5, 6 will break the optimization.
    # preconditioner type 3, 4, 5, 6 will break the optimization. 
    # these preconditioners are tries to only inverse $\rho_R$ , and handle the M, Minv by some linear map
    # There is no guarantee that the preconditioned gradient is a descent direction (preconditioner is not positive definite). so they don't work. 

    #function _precondition2(x::OptimState{MultiBosonCMPSData_MDMinv{T}}, dψ::MultiBosonCMPSData_MDMinv_Grad) where T
    #    ψ = x.data
    #    χ, d = get_χ(ψ), get_d(ψ)

    #    ϵ = isnan(x.df) ? fϵ(norm(dψ)^2) : fϵ(x.df)
    #    ϵ = max(1e-12, ϵ)
    #    PG = precondition_map_1(ψ, dψ; ϵ = ϵ)

    #    return PG
    #end
    #function _precondition4(x::OptimState{MultiBosonCMPSData_MDMinv{T}}, dψ::MultiBosonCMPSData_MDMinv_Grad) where T
    #    ψ = x.data
    #    ρR = right_env(ψ)

    #    ϵ = isnan(x.df) ? fϵ(norm(dψ)^2) : fϵ(x.df)
    #    ϵ = max(1e-12, ϵ)
    #    PG = preconditioner_map(ψ, dψ; ρR = ρR, ϵ = ϵ)

    #    return PG
    #end
    #function _precondition5(x::OptimState{MultiBosonCMPSData_MDMinv{T}}, dψ::MultiBosonCMPSData_MDMinv_Grad) where T
    #    ψ = x.data
    #    ρR = right_env(ψ)

    #    ϵ = isnan(x.df) ? fϵ(norm(dψ)^2) : fϵ(x.df)
    #    ϵ = max(1e-12, ϵ)
    #    PG = tangent_map1(ψ, dψ; ρR = ρR, ϵ = ϵ)

    #    return PG
    #end
    #function _precondition6(x::OptimState{MultiBosonCMPSData_MDMinv{T}}, dψ::MultiBosonCMPSData_MDMinv_Grad) where T
    #    ψ = x.data
    #    ρR = right_env(ψ)

    #    ϵ = isnan(x.df) ? fϵ(norm(dψ)^2) : fϵ(x.df)
    #    ϵ = max(1e-12, ϵ)
    #    PG = preconditioner_map1(ψ, dψ; ρR = ρR, ϵ = ϵ)

    #    return PG
    #end


    #elseif preconditioner_type == 3
    #    @show "using preconditioner 2"
    #    precondition1 = _precondition2
    #elseif preconditioner_type == 4
    #    @show "using preconditioner 4"
    #    precondition1 = _precondition4
    #elseif preconditioner_type == 5
    #    @show "using preconditioner 5"
    #    precondition1 = _precondition5
    #elseif preconditioner_type == 6
    #    @show "using preconditioner 6"
    #    precondition1 = _precondition6



#function tangent_map1(ψ::MultiBosonCMPSData_MDMinv{T}, g::MultiBosonCMPSData_MDMinv_Grad{T}; ρR = nothing, ϵ = 1e-10) where T
#    if isnothing(ρR)
#        ρR = right_env(ψ)
#    end
#
#    Id = Matrix{eltype(ρR)}(I, size(ρR))
#    ρRinv = inv(ρR + ϵ * Id)
#
#    EL = ψ.Minv * ψ.Minv' + sqrt(ϵ) * Id
#    ER = ψ.M' * ρRinv * ψ.M + sqrt(ϵ) * Id
#    Ms = [EL * (g.X * D - D * g.X + dD) * ER for (dD, D) in zip(g.dDs, ψ.Ds)] 
#
#    X_mapped = sum([M * D' - D' * M for (M, D) in zip(Ms, ψ.Ds)])
#    dDs_mapped = Diagonal.(Ms)
#
#    return MultiBosonCMPSData_MDMinv_Grad(dDs_mapped, X_mapped)
#end

#function preconditioner_map1(ψ::MultiBosonCMPSData_MDMinv, g::MultiBosonCMPSData_MDMinv_Grad; ϵ = 1e-10, ρR = nothing)
#    if isnothing(ρR)
#        ρR = right_env(ψ)
#    end
#
#    Id = Matrix{eltype(ρR)}(I, size(ρR))
#    ρRinv = inv(ρR + ϵ * Id)
#
#    Ws_mapped = [ψ.M * (g.X * D - D * g.X + dD) * ψ.Minv * ρRinv for (dD, D) in zip(g.dDs, ψ.Ds)]
#    function _f(gx::MultiBosonCMPSData_MDMinv_Grad, ::Val{false})
#        return [ψ.M * (gx.X * D - D * gx.X + dD) * ψ.Minv for (dD, D) in zip(gx.dDs, ψ.Ds)]
#    end 
#    function _f(Ws::Vector{Matrix{T}}, ::Val{true}) where T
#        Cs = [ψ.M' * W * ψ.Minv' for W in Ws]
#        dDs_mapped = Diagonal.(Cs)
#        X_mapped = sum([M * D' - D' * M for (M, D) in zip(Ws, ψ.Ds)])
#
#        return MultiBosonCMPSData_MDMinv_Grad(dDs_mapped, X_mapped)
#    end
#    
#    χ = size(ψ.M, 1)
#    g_mapped, _ = lssolve(_f, Ws_mapped; verbosity = 1, tol=1e-12, maxiter=χ^3)
#    return g_mapped
#end

#function preconditioner_map2(ψ::MultiBosonCMPSData_MDMinv, g::MultiBosonCMPSData_MDMinv_Grad; ϵ = 1e-10, ρR = nothing)
#    if isnothing(ρR)
#        ρR = right_env(ψ)
#    end
#
#    Id = Matrix{eltype(ρR)}(I, size(ρR))
#    ρRinv = inv(ρR)
#
#    C0s = [(g.X * D - D * g.X + dD) for (dD, D) in zip(g.dDs, ψ.Ds)]
#    Cs = [ψ.M' * ψ.M * C0 * ψ.Minv * ρRinv * ψ.Minv' + ϵ * C0 for C0 in C0s]
#    dDs = Diagonal.(Cs)
#    X = projection_X(Cs, ψ.Ds)
#    return MultiBosonCMPSData_MDMinv_Grad(dDs, X)
#end

#function projection_X(offdiags::Vector{Matrix{T}}, Ds::Vector{Diagonal{T, Vector{T}}}) where T
#    # min ‖ sum([X * D - D * X - offdiag for (D, offdiag) in zip(Ds, offdiags)]) ‖
#    χ = size(Ds[1], 1)
#
#    projected_X = zeros(T, χ, χ)
#    for ix in 1:χ, iy in 1:χ
#        up = sum([dot(D[iy, iy] - D[ix, ix], offdiag[ix, iy]) for (D, offdiag) in zip(Ds, offdiags)])
#        dn = sum([dot(D[iy, iy] - D[ix, ix], D[iy, iy] - D[ix, ix]) for D in Ds])
#        projected_X[ix, iy] = (ix == iy) ? 0 : up / dn
#    end
#    residual = sum(norm(projected_X * D - D * projected_X - (offdiag - Diagonal(offdiag))) for (D, offdiag) in zip(Ds, offdiags))
#    @info "residual, $residual, norm of projected_X, $(norm(projected_X))"
#    return projected_X
#end