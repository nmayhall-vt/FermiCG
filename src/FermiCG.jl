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
#
#####################################

export ConfigString
export ElectronicInts 
export ElectronicProblem 
end
