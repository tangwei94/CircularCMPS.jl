mutable struct MultiBosonCMPSData_MCMinv{T<:Number} <: AbstractCMPSData
    Q::Matrix{T}
    M::Matrix{T}
    Minv::Matrix{T}
    Cs::Vector{Matrix{T}}
    α::Real
    function MultiBosonCMPSData_MCMinv(Q::Matrix{T}, M::Matrix{T}, Minv::Matrix{T}, Cs::Vector{Matrix{T}}, α::Real) where T
        return new{T}(Q, M, Minv, Cs, α)
    end
    function MultiBosonCMPSData_MCMinv{T}(Q::Matrix{T}, M::Matrix{T}, Minv::Matrix{T}, Cs::Vector{Matrix{T}}, α::Real) where T
        if !(M * Minv ≈ Matrix{T}(I, size(M))) 
            @warn "M * Minv not close to I"
            Minv = inv(M)
        end
        return new{T}(Q, M, Minv, Cs, α)
    end
end
function MultiBosonCMPSData_MCMinv(; Q::Matrix{T}, M::Matrix{T}, Minv::Matrix{T}, Cs::Vector{Matrix{T}}, α::Real) where T
    return MultiBosonCMPSData_MCMinv{T}(Q, M, Minv, Cs, α)
end
function MultiBosonCMPSData_MCMinv(Q::Matrix{T}, M::Matrix{T}, Cs::Vector{Matrix{T}}, α::Real) where T
    Minv = inv(M)
    return MultiBosonCMPSData_MCMinv{T}(Q, M, Minv, Cs, α)
end
function MultiBosonCMPSData_MCMinv(f, χ::Integer, d::Integer, α::Real)
    Q = f(ComplexF64, χ, χ)
    M = f(ComplexF64, χ, χ)
    Cs = map(ix -> f(ComplexF64, χ, χ), 1:d)
    return MultiBosonCMPSData_MCMinv(Q, M, Cs, α)
end

# operations on the data. not on the cMPS. The data lives in a linear parameter space
Base.:+(ψ::MultiBosonCMPSData_MCMinv, ϕ::MultiBosonCMPSData_MCMinv) = MultiBosonCMPSData_MCMinv(ψ.Q + ϕ.Q, ψ.M + ϕ.M, ψ.Cs + ϕ.Cs, ψ.α)
Base.:-(ψ::MultiBosonCMPSData_MCMinv, ϕ::MultiBosonCMPSData_MCMinv) = MultiBosonCMPSData_MCMinv(ψ.Q - ϕ.Q, ψ.M - ϕ.M, ψ.Cs - ϕ.Cs, ψ.α)
Base.:*(ψ::MultiBosonCMPSData_MCMinv, x::Number) = MultiBosonCMPSData_MCMinv{ComplexF64}(ψ.Q * x, ψ.M * x, ψ.Minv / x, ψ.Cs * x, ψ.α)
Base.:*(x::Number, ψ::MultiBosonCMPSData_MCMinv) = MultiBosonCMPSData_MCMinv{ComplexF64}(ψ.Q * x, ψ.M * x, ψ.Minv / x, ψ.Cs * x, ψ.α)
Base.eltype(ψ::MultiBosonCMPSData_MCMinv) = eltype(ψ.Q)
LinearAlgebra.dot(ψ1::MultiBosonCMPSData_MCMinv, ψ2::MultiBosonCMPSData_MCMinv) = dot(ψ1.Q, ψ2.Q) + dot(ψ1.M, ψ2.M) + sum(dot.(ψ1.Cs, ψ2.Cs))
LinearAlgebra.norm(ψ::MultiBosonCMPSData_MCMinv) = sqrt(norm(dot(ψ, ψ)))

function Base.similar(ψ::MultiBosonCMPSData_MCMinv) 
    Q = similar(ψ.Q)
    Cs = similar.(ψ.Cs)
    return MultiBosonCMPSData_MCMinv(Q, copy(ψ.M), copy(ψ.Minv), Cs, ψ.α)
end
function randomize!(ψ::MultiBosonCMPSData_MCMinv)
    T = eltype(ψ)
    map!(x -> rand(T), ψ.Q, ψ.Q)
    map!(x -> rand(T), ψ.M, ψ.M)
    ψ.Minv .= inv(ψ.M)
    for ix in eachindex(ψ.Cs)
        c = (@view ψ.Cs[ix][:, :])
        map!(x -> rand(T), c, c)
    end
    return ψ
end

@inline get_χ(ψ::MultiBosonCMPSData_MCMinv) = size(ψ.Q, 1)
@inline get_d(ψ::MultiBosonCMPSData_MCMinv) = length(ψ.Cs)
TensorKit.space(ψ::MultiBosonCMPSData_MCMinv) = ℂ^(get_χ(ψ))

function CMPSData(ψ::MultiBosonCMPSData_MCMinv)
    χ, d = get_χ(ψ), get_d(ψ)
    
    Q = TensorMap(ψ.Q, ℂ^χ, ℂ^χ)
    Rs = map(ψ.Cs) do C
        C_diag = Diagonal(C)
        C_offdiag = (C - C_diag)
        C_modulated = C_diag + ψ.α * C_offdiag / sqrt(tr(C_offdiag' * C_offdiag))
        TensorMap(ψ.M * C_modulated * ψ.Minv, ℂ^χ, ℂ^χ)
    end
    return CMPSData(Q, Rs)
end

#function MultiBosonCMPSData_MCMinv(ψ::CMPSData)
#    Q = ψ.Q.data
#    χ = get_χ(ψ)
#    M = Matrix{ComplexF64}(I, χ, χ)
#    Minv = Matrix{ComplexF64}(I, χ, χ)
#    Ds = map(ix-> Diagonal(zeros(ComplexF64, χ)), 1:get_d(ψ))
#    Cs = map(R->R.data, ψ.Rs)
#    α = 1
#    return MultiBosonCMPSData_MCMinv(Q, M, Minv, Ds, Cs, α)
#end
function ChainRulesCore.rrule(::Type{CMPSData}, ψ::MultiBosonCMPSData_MCMinv)
    M, Minv = ψ.M, ψ.Minv
    Cs = ψ.Cs
    α = ψ.α
    function CMPSData_pushback(∂ψ)
        ∂Q = ∂ψ.Q.data
        Cs_diag = map(C -> Diagonal(C), Cs)
        Cs_offdiag = map(C -> C - Diagonal(C), Cs)
        Cs_modulated = map((C_diag, C_offdiag) -> C_diag + α * C_offdiag / sqrt(tr(C_offdiag' * C_offdiag)), Cs_diag, Cs_offdiag)

        ∂M = sum([∂R.data * Minv' * C_modulated' for (∂R, C_modulated) in zip(∂ψ.Rs, Cs_modulated)]) - 
             sum([Minv' * C_modulated' * M' * ∂R.data * Minv' for (∂R, C_modulated) in zip(∂ψ.Rs, Cs_modulated)])
        
        ∂Cs_diag = map(∂R -> Diagonal(M' * ∂R.data * Minv'), ∂ψ.Rs)
        
        OffDiagonal(x) = x - Diagonal(x)
        ∂Cs_offdiag = map((∂R, C_offdiag) -> α * OffDiagonal(M' * ∂R.data * Minv') / sqrt(tr(C_offdiag' * C_offdiag)) - α * 0.5 * (tr(C_offdiag' * C_offdiag) ^ (-3/2)) * 2*C_offdiag * real(tr(∂R.data * Minv' * C_offdiag' * M')), ∂ψ.Rs, Cs_offdiag) 

        ∂Cs = ∂Cs_diag .+ ∂Cs_offdiag

        return NoTangent(), MultiBosonCMPSData_MCMinv(∂Q, ∂M, ∂Cs, α) 
    end
    return CMPSData(ψ), CMPSData_pushback
end

function left_canonical(ψ::MultiBosonCMPSData_MCMinv)
    ψc = CMPSData(ψ)

    X, ψcl = left_canonical(ψc)
    Q = ψcl.Q.data 
    M = X.data * ψ.M 
    Minv = ψ.Minv * inv(X.data)

    return MultiBosonCMPSData_MCMinv(Q, M, Minv, deepcopy(ψ.Cs), ψ.α)
end
function right_env(ψ::MultiBosonCMPSData_MCMinv)
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

function retract_left_canonical(ψ::MultiBosonCMPSData_MCMinv{T}, α::Float64, dCs::Vector{Matrix{T}}, X::Matrix{T}) where T
    # check left canonical form 
    ψc = CMPSData(ψ)
    ϵ = norm(ψc.Q + ψc.Q' + sum([R' * R for R in ψc.Rs]))
    #(ϵ > 1e-9) && @warn "your cmps has deviated from the left canonical form, err=$ϵ"

    Cs = ψ.Cs .+ α .* dCs

    #X[diagind(X)] .- tr(X) / size(X, 1) # make X traceless
    M = exp(α * X) * ψ.M
    Minv = ψ.Minv * exp(-α * X)

    fC(C) = Diagonal(C) + ψ.α * (C - Diagonal(C)) / sqrt(tr(C' * C))
    Rs = [ψ.M * fC(C0) * ψ.Minv for C0 in ψ.Cs] 
    R1s = [M * fC(C) * Minv for C in Cs] 
    ΔRs = R1s .- Rs

    Q = ψ.Q - sum([R' * ΔR + 0.5 * ΔR' * ΔR for (R, ΔR) in zip(Rs, ΔRs)])

    return MultiBosonCMPSData_MCMinv(Q, M, Minv, Cs, ψ.α)
end

#function expand(ψ::MultiBosonCMPSData_MCMinv, χ::Integer; perturb::Float64=1e-3)
#    χ0, d = get_χ(ψ), get_d(ψ)
#    if χ <= χ0
#        @warn "new χ not bigger than χ0"
#        return ψ
#    end
#
#    Qd, Qu = eigen(ψ.Q)
#    _, Qminind = findmin(real.(Qd))
#    Q = diagm(vcat(Qd, fill(Qd[Qminind] - log(10), χ - χ0)))
#
#    # the norm of M and Minv can be very ill-conditioned, e.g., one of them is very small and the other is very large
#    α = norm(ψ.M) / norm(ψ.Minv)
#    M0 = ψ.M / sqrt(α)
#
#    M = Matrix{eltype(ψ)}(I, χ, χ)
#    M += rand(eltype(ψ), χ, χ) * perturb
#    M[1:χ0, 1:χ0] = inv(Qu) * M0
#
#    Ds = map(1:d) do ix
#        Diagonal(vcat(diag(ψ.Ds[ix]), fill(perturb, 1:χ-χ0)))
#    end
#
#    return MultiBosonCMPSData_MCMinv(Q, M, Ds) 
#end
#
struct MultiBosonCMPSData_MCMinv_Grad{T<:Number} <: AbstractCMPSData
    dCs::Vector{Matrix{T}}
    X::Matrix{T}
end
function MultiBosonCMPSData_MCMinv_Grad(v::Vector{T}, χ::Int, d::Int) where T
    dCs = map(ix -> reshape(v[(χ^2)*(ix-1)+1:(χ^2)*ix], (χ, χ)), 1:d)
    X = reshape(v[d*χ^2+1:end], χ, χ)
    return MultiBosonCMPSData_MCMinv_Grad{T}(dCs, X)
end

Base.:+(a::MultiBosonCMPSData_MCMinv_Grad, b::MultiBosonCMPSData_MCMinv_Grad) = MultiBosonCMPSData_MCMinv_Grad(a.dCs .+ b.dCs, a.X + b.X)
Base.:-(a::MultiBosonCMPSData_MCMinv_Grad, b::MultiBosonCMPSData_MCMinv_Grad) = MultiBosonCMPSData_MCMinv_Grad(a.dCs .- b.dCs, a.X - b.X)
Base.:*(a::MultiBosonCMPSData_MCMinv_Grad, x::Number) = MultiBosonCMPSData_MCMinv_Grad(a.dCs * x, a.X * x)
Base.:*(x::Number, a::MultiBosonCMPSData_MCMinv_Grad) = MultiBosonCMPSData_MCMinv_Grad(a.dCs * x, a.X * x)
Base.eltype(a::MultiBosonCMPSData_MCMinv_Grad) = eltype(a.X)
LinearAlgebra.dot(a::MultiBosonCMPSData_MCMinv_Grad, b::MultiBosonCMPSData_MCMinv_Grad) = sum(dot.(a.dCs, b.dCs)) + dot(a.X, b.X)
TensorKit.inner(a::MultiBosonCMPSData_MCMinv_Grad, b::MultiBosonCMPSData_MCMinv_Grad) = real(dot(a, b))
LinearAlgebra.norm(a::MultiBosonCMPSData_MCMinv_Grad) = sqrt(norm(dot(a, a)))
Base.similar(a::MultiBosonCMPSData_MCMinv_Grad) = MultiBosonCMPSData_MCMinv_Grad(similar.(a.dCs), similar(a.X))
Base.vec(a::MultiBosonCMPSData_MCMinv_Grad) = vcat(vec.(a.dCs)..., vec(a.X))

function randomize!(a::MultiBosonCMPSData_MCMinv_Grad)
    T = eltype(a)
    for ix in eachindex(a.dCs)
        map!(x -> rand(T), a.dCs[ix], a.dCs[ix])
    end
    map!(x -> rand(T), a.X, a.X)
    return a
end

function diff_to_grad(ψ::MultiBosonCMPSData_MCMinv, ∂ψ::MultiBosonCMPSData_MCMinv)
    Rs = map(C->ψ.M * C * ψ.Minv, ψ.Cs)

    gCs = [-ψ.M' * R * ∂ψ.Q * ψ.Minv' + ∂C for (R, ∂C) in zip(Rs, ∂ψ.Cs)]
    gX = sum([- R * ∂ψ.Q * R' + R' * R * ∂ψ.Q for R in Rs]) + ∂ψ.M * ψ.M'
    #gX[diagind(gX)] .-= tr(gX) / size(gX, 1) # make X traceless
    return MultiBosonCMPSData_MCMinv_Grad(gCs, gX)
end

function tangent_map(ψ::MultiBosonCMPSData_MCMinv, g::MultiBosonCMPSData_MCMinv_Grad; ρR = nothing)
    if isnothing(ρR)
        ρR = right_env(ψ)
    end

    fC(C) = Diagonal(C) + ψ.α * (C - Diagonal(C)) / sqrt(tr(C' * C))
    Rs = map(C->ψ.M * fC(C) * ψ.Minv, ψ.Cs) 

    X = sum(map(zip(g.dCs, Rs)) do (dC, R)
            g.X * R * ρR * R' - R' * g.X * R * ρR -
            R * g.X * ρR * R' + R' * R * g.X * ρR + 
            ψ.M * dC * ψ.Minv * ρR * R' - R' * ψ.M * dC * ψ.Minv * ρR # FIXME
        end)

    dCs = map(zip(g.dCs, Rs)) do (dC, R)
        ψ.M' * g.X * R * ρR * ψ.Minv' -
            ψ.M' * R * g.X * ρR * ψ.Minv' + 
            ψ.M' * ψ.M * dC * ψ.Minv * ρR * ψ.Minv'
    end

    return MultiBosonCMPSData_MCMinv_Grad(dCs, X)
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