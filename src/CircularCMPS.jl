module CircularCMPS

# TODO. 
# - lazy evaluation for backward gradients 
# - struct "optimization_state", including the cMPS and environments etc
# - symmetries

__precompile__(true)

using Parameters
using LinearAlgebra
using VectorInterface, TensorKit, TensorOperations, KrylovKit, TensorKitManifolds
using ChainRules, ChainRulesCore, Zygote, FiniteDifferences
using OptimKit
using LoopVectorization 
using Tullio
using JLD2
using Printf
#using FastGaussQuadrature

# utils.jl
export MPSBondTensor, GenericMPSTensor, MPSTensor, fill_data!, randomize!, K_permute, K_permute_back, K_otimes, Kact_R, Kact_L, herm_reg_inv, quasi_inv 

# cmps.jl
export AbstractCMPSData, CMPSData, get_χ, get_d, get_matrices, transfer_matrix, transfer_matrix_dagger, left_canonical, right_canonical, expand, K_mat, finite_env, rescale

# operators.jl
export kinetic, particle_density, point_interaction, pairing

# ground_state.jl 
export lieb_liniger_ground_state

# excited_state.jl 
export θ2, θ3, AbstractCoeffs, Coeff2, Coeff3, ExcitationData, gauge_fixing_map, effective_N, effective_H, Kac_Moody_gen

# transfer_matrix.jl
export TransferMatrix, left_env, right_env, Kmat_pseudo_inv

# multi_boson_cmps.jl
export MultiBosonCMPSData, tangent_map

# multi_boson_cmps_MDMinv.jl
export MultiBosonCMPSData_MDMinv, MultiBosonCMPSData_MDMinv_Grad

# multi_boson_cmps_tnp.jl
export MultiBosonCMPSData_tnp, MultiBosonCMPSData_tnp_Grad

# multi_boson_cmps_MCMinv.jl
#export MultiBosonCMPSData_MCMinv#, MultiBosonCMPSData_MCMinv_Grad

# multi_boson_cmps_P.jl
export MultiBosonCMPSData_P

# optim_alg.jl 
export CircularCMPSRiemannian, minimize, leading_boundary_cmps

# cmpo.jl 
export AbstractCMPO, CMPO, ln_ovlp, compress, direct_sum, W_mul, variance, free_energy, energy, klein

# cmpo_zoo.jl
export ising_cmpo, xxz_af_cmpo, xxz_fm_cmpo, heisenberg_j1j2_cmpo, rydberg_cmpo

# entanglement.jl 
export half_chain_singular_values, entanglement_entropy

# power_iteration.jl 
export PowerMethod, power_iteration

# variational_optim.jl 
export VariationalOptim, leading_boundary

# hamiltonian_zoo.jl
export AbstractHamiltonian, SingleBosonLiebLiniger, MultiBosonLiebLiniger, ground_state, MultiBosonLiebLinigerWithPairing

# cMPS code for continuous Hamiltonians
include("utils.jl");
include("cmps.jl");
include("cmpsAD.jl");
include("operators.jl");
include("ground_state.jl")
include("excited_state.jl");
include("optim_alg.jl");
include("transfer_matrix.jl");
include("multiple_bosons/hamiltonian_zoo.jl");
include("multiple_bosons/multi_boson_cmps_via_penalty.jl");
include("multiple_bosons/multi_boson_cmps_diag.jl");
include("multiple_bosons/multi_boson_cmps_tnp.jl");
include("multiple_bosons/multi_boson_cmps_MBMinv.jl");
include("multiple_bosons/multi_boson_cmps_MDMinv.jl");
#include("multiple_bosons/multi_boson_cmps_MCMinv.jl");

# cMPO code
include("cmpo.jl");
include("cmpoAD.jl");
include("cmpo_zoo.jl");
include("entanglement.jl");
include("power_iteration.jl")
include("power_iteration_1.jl")
include("variational_optim.jl")

end
