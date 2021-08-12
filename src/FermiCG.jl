"""
General electronic course-graining platform
"""
module FermiCG


#####################################
# External packages
#
using Compat
using HDF5
using KrylovKit
using LinearAlgebra
using NDTensors
using PackageCompiler
using Parameters
using Printf
using TimerOutputs
using BenchmarkTools 
using OrderedCollections 
using IterTools
using LinearAlgebra
using StaticArrays
using TensorOperations

using ThreadPools
using Distributed
using JLD2
using LinearMaps
# using Unicode
#
#####################################



#####################################
# Local Imports
#
include("Utils.jl")
include("hosvd.jl")
include("StringCI/StringCI.jl");
include("SymDenseMats.jl");
include("Solvers.jl");

# Problem definition stuff
include("type_Atom.jl");
include("type_Molecule.jl");
include("type_InCoreInts.jl");

# Local data
include("type_Cluster.jl")
include("type_ClusterOps.jl")
include("type_ClusterBasis.jl")
include("type_ClusterSubspace.jl")
include("build_local_quantities.jl")

#indexing
const FockIndex = Tuple{Int16, Int16}
include("type_SparseIndex.jl")
include("type_ClusterConfig.jl")
include("type_TransferConfig.jl")
include("type_FockConfig.jl")
include("type_TuckerConfig.jl")
include("type_OperatorConfig.jl")
include("Indexing.jl")

include("type_AbstractState.jl")
include("type_BSstate.jl")
include("type_BSTstate.jl")
include("type_TPSCIstate.jl")

include("type_ClusteredTerm.jl")
include("type_ClusteredOperator.jl")
#include("TPSCIstates.jl")

include("tucker_inner.jl")
include("tucker_outer.jl")
include("bst.jl")

include("tpsci_inner.jl")
include("tpsci_matvec_thread.jl")
include("tpsci_pt1_wavefunction.jl")
include("tpsci_pt2_energy.jl")
include("tpsci_outer.jl")
include("tpsci_helpers.jl")

include("dense_inner.jl")
include("dense_outer.jl")

include("CMFs.jl")
include("pyscf/PyscfFunctions.jl");
#
#####################################

export StringCI
export InCoreInts
export Molecule
export Atom
export Cluster
export ClusterBasis
export TPSCIstate
export ClusterConfig 
export FockConfig 

end
