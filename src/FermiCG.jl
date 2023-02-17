"""
General electronic course-graining platform
"""
module FermiCG


#####################################
# External packages
#
using Compat
using KrylovKit
using LinearAlgebra
using Printf
using TimerOutputs
using OrderedCollections 
using IterTools
using LinearAlgebra
using StaticArrays
using TensorOperations

using ThreadPools
using Distributed
using JLD2
using LinearMaps

# our packages
using QCBase
using InCoreIntegrals
using RDM
using BlockDavidson 
using ActiveSpaceSolvers

# using Unicode
#
#####################################

#####################################
# Local Imports
#
include("Utils.jl")
include("hosvd.jl")
include("SymDenseMats.jl");

# Local data
include("type_ClusterOps.jl")
include("type_ClusterBasis.jl")
include("type_ClusterSubspace.jl")

#indexing
include("type_SparseIndex.jl")
include("type_ClusterConfig.jl")
include("type_TransferConfig.jl")
include("type_FockConfig.jl")
include("type_TuckerConfig.jl")
include("type_OperatorConfig.jl")
include("Indexing.jl")
include("build_local_quantities.jl")

include("type_AbstractState.jl")
include("type_BSstate.jl")
include("type_BSTstate.jl")
include("type_TPSCIstate.jl")

include("type_ClusteredTerm.jl")
include("type_ClusteredOperator.jl")
#include("TPSCIstates.jl")

include("tucker_inner.jl")
include("tucker_build_dense_H_term.jl")
include("tucker_contract_dense_H_with_state.jl")
include("tucker_form_sigma_block_expand.jl")
include("tucker_outer.jl")
include("tucker_pt.jl")
include("bst.jl")

include("tpsci_inner.jl")
include("tpsci_matvec_thread.jl")
include("tpsci_pt1_wavefunction.jl")
include("tpsci_pt2_energy.jl")
include("tpsci_outer.jl")
include("tpsci_helpers.jl")

include("dense_inner.jl")
include("dense_outer.jl")

#
#####################################

export RDM 
export InCoreInts
export Molecule
export Atom
export MOCluster
export ClusterBasis
export ClusterSubspace
export ClusteredOperator
export TPSCIstate
export BSTstate
export ClusterConfig 
export FockConfig 
export TuckerConfig 
export n_orb
export add_subspace!
export add_fockconfig!
export expand_each_fock_space!
export block_sparse_tucker
end
