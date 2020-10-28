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
#
#####################################



#####################################
# Local Imports
#
include("Tools.jl")
include("Hamiltonians.jl")
include("ConfigStrings.jl")
include("pyscf/PyscfFunctions.jl")
#
#####################################

export ConfigString
export ElectronicInts
#export ElectronicProblem
export Molecule
export Atom
#export get_pyscf_integrals
#export pyscf_fci
end
