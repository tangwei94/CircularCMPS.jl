using TensorKit, LinearAlgebra, KrylovKit
using ChainRules, Zygote 
using QuadGK
using Test
using Revise
using CircularCMPS

function test_ADgrad(_F, X)

        # retraction direction
        sX = similar(X)
        randomize!(sX)

        # finite diff along retraction direction
        α = 1e-5
        ∂α1 = (_F(X + α * sX) - _F(X - α * sX)) / (2 * α)
        α = 1e-6
        ∂α2 = (_F(X + α * sX) - _F(X - α * sX)) / (2 * α)

        # test correctness of derivative from AD
        ∂X = _F'(X);
        ∂αad = real(dot(∂X, sX))
        @test abs(∂α1 - ∂αad) < 1e-5
        @test abs(∂α2 - ∂αad) < 1e-6

end

include("test_cmps.jl");
include("test_excitation.jl")
include("test_transfer_matrix.jl")
include("test_multi_boson_cmps.jl")
