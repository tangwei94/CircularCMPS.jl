mutable struct MultiBosonCMPSData_MDMinv{T<:Number} <: AbstractCMPSData
    Q::Matrix{T}
    M::Matrix{T}
    Minv::Matrix{T}
    Ds::Vector{Diagonal{T, Vector{T}}}
    function MultiBosonCMPSData_MDMinv(Q::Matrix{T}, M::Matrix{T}, Minv::Matrix{T}, Ds::Vector{Diagonal{T, Vector{T}}}) where T
        if !(M * Minv ≈ Matrix{T}(I, size(M))) 
            @warn "M * Minv not close to I"
            Minv = inv(M)
        end
        return new{T}(Q, M, Minv, Ds)
    end
    function MultiBosonCMPSData_MDMinv(Q::Matrix{T}, M::Matrix{T}, Ds::Vector{Diagonal{T, Vector{T}}}) where T
        Minv = inv(M)
        return new{T}(Q, M, Minv, Ds)
    end
    function MultiBosonCMPSData_MDMinv(f, χ::Integer, d::Integer)
        Q = f(ComplexF64, χ, χ)
        M, _ = qr(f(ComplexF64, χ, χ))

        Minv = M'
        Ds = map(ix -> Diagonal(randn(ComplexF64, χ)), 1:d)
        return new{ComplexF64}(Q, M, Minv, Ds)
    end
end

# operations on the data. not on the cMPS
Base.:+(ψ::MultiBosonCMPSData_MDMinv, ϕ::MultiBosonCMPSData_MDMinv) = MultiBosonCMPSData_MDMinv(ψ.Q + ϕ.Q, ψ.M + ϕ.M, ψ.Ds + ϕ.Ds)
Base.:-(ψ::MultiBosonCMPSData_MDMinv, ϕ::MultiBosonCMPSData_MDMinv) = MultiBosonCMPSData_MDMinv(ψ.Q - ϕ.Q, ψ.M - ϕ.M, ψ.Ds - ϕ.Ds)
Base.:*(ψ::MultiBosonCMPSData_MDMinv, x::Number) = MultiBosonCMPSData_MDMinv(ψ.Q * x, ψ.M * x, ψ.Minv / x, ψ.Ds * x)
Base.:*(x::Number, ψ::MultiBosonCMPSData_MDMinv) = MultiBosonCMPSData_MDMinv(ψ.Q * x, ψ.M * x, ψ.Minv / x, ψ.Ds * x)
Base.eltype(ψ::MultiBosonCMPSData_MDMinv) = eltype(ψ.Q)
LinearAlgebra.dot(ψ1::MultiBosonCMPSData_MDMinv, ψ2::MultiBosonCMPSData_MDMinv) = dot(ψ1.Q, ψ2.Q) + dot(ψ1.M, ψ2.M) + sum(dot.(ψ1.Ds, ψ2.Ds))
LinearAlgebra.norm(ψ::MultiBosonCMPSData_MDMinv) = sqrt(norm(dot(ψ, ψ)))

function Base.similar(ψ::MultiBosonCMPSData_MDMinv) 
    Q = similar(ψ.Q)
    Ds = similar.(ψ.Ds)
    return MultiBosonCMPSData_MDMinv(Q, copy(ψ.M), copy(ψ.Minv), Ds)
end
function randomize!(ψ::MultiBosonCMPSData_MDMinv)
    T = eltype(ψ)
    map!(x -> rand(T), ψ.Q, ψ.Q)
    map!(x -> rand(T), ψ.M, ψ.M)
    ψ.Minv .= inv(ψ.M)
    for ix in eachindex(ψ.Ds)
        d = view(ψ.Ds[ix], diagind(ψ.Ds[ix]))
        map!(x -> rand(T), d, d)
    end
    return ψ
end

@inline get_χ(ψ::MultiBosonCMPSData_MDMinv) = size(ψ.Q, 1)
@inline get_d(ψ::MultiBosonCMPSData_MDMinv) = length(ψ.Ds)
TensorKit.space(ψ::MultiBosonCMPSData_MDMinv) = ℂ^(get_χ(ψ))

function CMPSData(ψ::MultiBosonCMPSData_MDMinv)
    χ, d = get_χ(ψ), get_d(ψ)
    
    Q = TensorMap(ψ.Q, ℂ^χ, ℂ^χ)
    Rs = map(ψ.Ds) do D 
        TensorMap(ψ.M * D * ψ.Minv, ℂ^χ, ℂ^χ)
    end
    return CMPSData(Q, Rs)
end

function MultiBosonCMPSData_MDMinv(ψ::CMPSData)
    @warn "MultiBosonCMPSData_MDMinv(ψ::CMPSData) is going to be removed"
    Q = ψ.Q.data
    _, M = eigen(ψ.Rs[1].data)
    Minv = inv(M)
  
    D0s = map(R->Minv * R.data * M, ψ.Rs)
    Ds = map(D->Diagonal(D), D0s)
    err2 = sum(norm.(Ds .- D0s) .^ 2)
    @info "convert CMPSData to MultiBosonCMPSData_MDMinv, err = $(sqrt(err2))"
    return MultiBosonCMPSData_MDMinv(Q, M, Minv, Ds)
end
function convert_to_MultiBosonCMPSData_MDMinv_deprecated(ψ::CMPSData)
    Q = ψ.Q.data
    _, M = eigen(ψ.Rs[1].data)
    Minv = inv(M)
  
    D0s = map(R->Minv * R.data * M, ψ.Rs)
    Ds = map(D->Diagonal(D), D0s)
    err2 = sum(norm.(Ds .- D0s) .^ 2)
    
    return MultiBosonCMPSData_MDMinv(Q, M, Minv, Ds), sqrt(err2)
end

function convert_to_MultiBosonCMPSData_MDMinv(ψ::CMPSData)

    # FIXME. a temporary solution; only works for the case of two species of bosons
    R1, R2 = ψ.Rs[1].data, ψ.Rs[2].data
    function fv(v::Vector{Float64})
        α, β = v
        _, V1 = eigen(R1 + (α + im*β) * R2);
        invV1 = inv(V1)
        D1 = (inv(V1) * R1 * V1)
        D2 = (inv(V1) * R2 * V1)

        err2(D) = norm(V1 * (D - Diagonal(D)) * invV1) ^ 2
        y = sqrt(err2(D1) + err2(D2))
        return y
    end

    function fgv(v::Vector{Float64})
        y = fv(v)
        g = grad(central_fdm(5, 1), fv, v)[1]
        return y, g
    end

    ymin, vmin = Inf, [0, 0]
    v0 = [0.0, 0.0]
    for _ in 1:10
        v0 += 2 * rand(2) .- 1
        res = optimize(fgv, v0, LBFGS(;verbosity=0, gradtol=1e-4))
        if res[2] < ymin
            vmin = res[1]
            ymin = res[2]
        end
    end

    α, β = vmin
    _, M = eigen(R1 + (α + im*β) * R2);

    Q = ψ.Q.data
    _, M = eigen(ψ.Rs[1].data)
    Minv = inv(M)
  
    D0s = map(R->Minv * R.data * M, ψ.Rs)
    Ds = map(D->Diagonal(D), D0s)
    err2 = sum(norm.(Ds .- D0s) .^ 2)
    
    return MultiBosonCMPSData_MDMinv(Q, M, Minv, Ds), sqrt(err2)
end

function ChainRulesCore.rrule(::Type{CMPSData}, ψ::MultiBosonCMPSData_MDMinv)
    M, Minv = ψ.M, ψ.Minv
    Ds = ψ.Ds
    function CMPSData_pushback(∂ψ)
        ∂Q = ∂ψ.Q.data
        ∂Ds = map(∂R -> Diagonal(M' * ∂R.data * Minv'), ∂ψ.Rs)
        ∂M = sum([∂R.data * Minv' * D' for (∂R, D) in zip(∂ψ.Rs, Ds)]) - 
             sum([Minv' * D' * M' * ∂R.data * Minv' for (∂R, D) in zip(∂ψ.Rs, Ds)])
        return NoTangent(), MultiBosonCMPSData_MDMinv(∂Q, ∂M, ∂Ds) 
    end
    return CMPSData(ψ), CMPSData_pushback
end

function left_canonical(ψ::MultiBosonCMPSData_MDMinv)
    ψc = CMPSData(ψ)

    X, ψcl = left_canonical(ψc)
    Q = ψcl.Q.data 
    M = X.data * ψ.M 
    Minv = ψ.Minv * inv(X.data)

    return MultiBosonCMPSData_MDMinv(Q, M, Minv, deepcopy(ψ.Ds))
end
function right_env(ψ::MultiBosonCMPSData_MDMinv)
    # transfer matrix
    ψc = CMPSData(ψ)
    fK = transfer_matrix(ψc, ψc)
    
    # solve the fixed-point equation
    init = similar(ψc.Q, space(ψc.Q, 1)←space(ψc.Q, 1))
    randomize!(init);
    _, vls, _ = eigsolve(fK, init, 1, :LR)
    vl = vls[1]
    
    U, S, _ = svd(vl.data)
    return U * Diagonal(S) * U'
end

function retract_left_canonical(ψ::MultiBosonCMPSData_MDMinv{T}, α::Float64, dDs::Vector{Diagonal{T, Vector{T}}}, X::Matrix{T}) where T
    # check left canonical form 
    ψc = CMPSData(ψ)
    ϵ = norm(ψc.Q + ψc.Q' + sum([R' * R for R in ψc.Rs]))
    (ϵ > 1e-10) && @warn "your cmps has deviated from the left canonical form, err=$ϵ"

    Ds = ψ.Ds .+ α .* dDs
    #X[diagind(X)] .- tr(X) / size(X, 1) # make X traceless
    M = exp(α * X) * ψ.M
    Minv = ψ.Minv * exp(-α * X)

    Rs = [ψ.M * D0 * ψ.Minv for D0 in ψ.Ds] 
    R1s = [M * D * Minv for D in Ds] 
    ΔRs = R1s .- Rs

    Q = ψ.Q - sum([R' * ΔR + 0.5 * ΔR' * ΔR for (R, ΔR) in zip(Rs, ΔRs)])

    return MultiBosonCMPSData_MDMinv(Q, M, Minv, Ds)
end

function expand(ψ::MultiBosonCMPSData_MDMinv, χ::Integer; perturb::Float64=1e-3)
    χ0, d = get_χ(ψ), get_d(ψ)
    if χ <= χ0
        @warn "new χ not bigger than χ0"
        return ψ
    end

    Qd, Qu = eigen(ψ.Q)
    _, Qminind = findmin(real.(Qd))
    Q = diagm(vcat(Qd, fill(Qd[Qminind] - log(10), χ - χ0)))

    # the norm of M and Minv can be very ill-conditioned, e.g., one of them is very small and the other is very large
    α = norm(ψ.M)/norm(ψ.Minv)
    M0 = ψ.M / sqrt(α)

    M = Matrix{eltype(ψ)}(I, χ, χ)
    M += rand(eltype(ψ), χ, χ) * perturb
    M[1:χ0, 1:χ0] = inv(Qu) * M0

    Ds = map(1:d) do ix
        Diagonal(vcat(diag(ψ.Ds[ix]), fill(perturb, 1:χ-χ0)))
    end

    return MultiBosonCMPSData_MDMinv(Q, M, Ds) 
end

struct MultiBosonCMPSData_MDMinv_Grad{T<:Number} <: AbstractCMPSData
    dDs::Vector{Diagonal{T, Vector{T}}}
    X::Matrix{T}
    function MultiBosonCMPSData_MDMinv_Grad(dDs::Vector{Diagonal{T, Vector{T}}}, X::Matrix{T}) where T
        return new{T}(dDs, X)
    end
    function MultiBosonCMPSData_MDMinv_Grad(v::Vector{T}, χ::Int, d::Int) where T
        dDs = map(ix -> Diagonal(v[χ*(ix-1)+1:χ*ix]), 1:d)
        X = reshape(v[χ*d+1:end], χ, χ)
        return new{T}(dDs, X)
    end
end

Base.:+(a::MultiBosonCMPSData_MDMinv_Grad, b::MultiBosonCMPSData_MDMinv_Grad) = MultiBosonCMPSData_MDMinv_Grad(a.dDs .+ b.dDs, a.X + b.X)
Base.:-(a::MultiBosonCMPSData_MDMinv_Grad, b::MultiBosonCMPSData_MDMinv_Grad) = MultiBosonCMPSData_MDMinv_Grad(a.dDs .- b.dDs, a.X - b.X)
Base.:*(a::MultiBosonCMPSData_MDMinv_Grad, x::Number) = MultiBosonCMPSData_MDMinv_Grad(a.dDs * x, a.X * x)
Base.:*(x::Number, a::MultiBosonCMPSData_MDMinv_Grad) = MultiBosonCMPSData_MDMinv_Grad(a.dDs * x, a.X * x)
Base.eltype(a::MultiBosonCMPSData_MDMinv_Grad) = eltype(a.X)
LinearAlgebra.dot(a::MultiBosonCMPSData_MDMinv_Grad, b::MultiBosonCMPSData_MDMinv_Grad) = sum(dot.(a.dDs, b.dDs)) + dot(a.X, b.X)
LinearAlgebra.norm(a::MultiBosonCMPSData_MDMinv_Grad) = sqrt(norm(dot(a, a)))
Base.similar(a::MultiBosonCMPSData_MDMinv_Grad) = MultiBosonCMPSData_MDMinv_Grad(similar.(a.dDs), similar(a.X))
Base.vec(a::MultiBosonCMPSData_MDMinv_Grad) = vcat(diag.(a.dDs)..., vec(a.X))

function randomize!(a::MultiBosonCMPSData_MDMinv_Grad)
    T = eltype(a)
    for ix in eachindex(a.dDs)
        v = view(a.dDs[ix], diagind(a.dDs[ix]))
        map!(x -> rand(T), v, v)
    end
    map!(x -> rand(T), a.X, a.X)
    return a
end

function diff_to_grad(ψ::MultiBosonCMPSData_MDMinv, ∂ψ::MultiBosonCMPSData_MDMinv)
    Rs = map(D->ψ.M * D * ψ.Minv, ψ.Ds)

    gDs = [Diagonal(-ψ.M' * R * ∂ψ.Q * ψ.Minv' + ∂D) for (R, ∂D) in zip(Rs, ∂ψ.Ds)]
    gX = ψ.M * sum([- R * ∂ψ.Q * R' + R' * R * ∂ψ.Q for R in Rs]) * ψ.Minv' + ∂ψ.M * ψ.M'
    #gX[diagind(gX)] .-= tr(gX) / size(gX, 1) # make X traceless
    return MultiBosonCMPSData_MDMinv_Grad(gDs, gX)
end

function tangent_map(ψ::MultiBosonCMPSData_MDMinv, g::MultiBosonCMPSData_MDMinv_Grad; ρR = nothing)
    if isnothing(ρR)
        ρR = right_env(ψ)
    end

    Rs = map(D->ψ.M * D * ψ.Minv, ψ.Ds)
    X1 = ψ.M * g.X * ψ.Minv

    X1_maped = sum(map(zip(g.dDs, Rs)) do (dD, R)
            X1 * R * ρR * R' - R' * X1 * R * ρR -
            R * X1 * ρR * R' + R' * R * X1 * ρR + 
            ψ.M * dD * ψ.Minv * ρR * R' - R' * ψ.M * dD * ψ.Minv * ρR
        end)
    X_mapped = ψ.Minv * X1_maped * ψ.M

    dDs_mapped = map(zip(g.dDs, Rs)) do (dD, R)
        Diagonal(ψ.M' * X1 * R * ρR * ψ.Minv' -
            ψ.M' * R * X1 * ρR * ψ.Minv' + 
            ψ.M' * ψ.M * dD * ψ.Minv * ρR * ψ.Minv')
    end

    return MultiBosonCMPSData_MDMinv_Grad(dDs_mapped, X_mapped)
end


#function tangent_map(ψm::MultiBosonCMPSData, Xm::MultiBosonCMPSData, EL::MPSBondTensor, ER::MPSBondTensor, Kinv::AbstractTensorMap{S, 2, 2}) where {S}
#    χ = get_χ(ψm)
#    ψ = CMPSData(ψm)
#    X = CMPSData(Xm)
#    Id = id(ℂ^χ)
#
#    ER /= tr(EL * ER)
#
#    K1 = K_permute(K_otimes(Id, X.Q) + sum(K_otimes.(ψ.Rs, X.Rs)))
#    @tensor ER1[-1; -2] := Kinv[-1 4; 3 -2] * K1[3 2; 1 4] * ER[1; 2]
#    @tensor EL1[-1; -2] := Kinv[4 -1; -2 3] * K1[2 3; 4 1] * EL[1; 2]
#    @tensor singular = EL[1; 2] * K1[2 3; 4 1] * ER[4; 3]
#
#    mapped_XQ = EL * ER1 + EL1 * ER + singular * EL * ER
#    mapped_XRs = map(zip(ψ.Rs, X.Rs)) do (R, XR)
#        EL * XR * ER + EL1 * R * ER + EL * R * ER1 + singular * EL * R * ER
#    end
#
#    return MultiBosonCMPSData(CMPSData(mapped_XQ, mapped_XRs)) 
#end