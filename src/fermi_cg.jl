"""
General electronic course-graining platform
"""
module fermi_cg


#####################################
# External packages
#
using Compat
using HDF5
using KrylovKit
using LinearAlgebra
using NDTensors
using PackageCompiler
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

end
