using Pkg
import DelimitedFiles: readdlm, writedlm
import Random.randperm
import Statistics: mean, std
import Printf.@printf

installed_pkg = keys(Pkg.installed())

# auto-install dependencies
!(      "PyPlot" ∈ installed_pkg ) && Pkg.add("PyPlot")
!( "ScikitLearn" ∈ installed_pkg ) && Pkg.add("ScikitLearn")
!( "MultivariateStats" ∈ installed_pkg ) && Pkg.add("MultivariateStats")



# for plotting
using PyPlot

# machine learning library
using ScikitLearn

# contains PCA
import MultivariateStats