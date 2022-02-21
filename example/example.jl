using Distributions
using IndependentHypothesisWeighting
using Random
using MLDataPattern
import MLDataPattern: kfolds

using StatsBase
using CategoricalArrays

Random.seed!(1)
m= 10000
kf = kfolds(MLDataPattern.shuffleobs(1:m), 5)
Xs = CategoricalVector(sample(1:2, m))
Ps = rand(BetaUniformMixtureModel(0.7, 0.2), m) .* (Xs.==1) .+ rand(Uniform(), m) .* (Xs.==2)

typeof(Xs)
typeof(Ps)

α = 0.1
sum(adjust(Ps, BenjaminiHochberg()) .<= α ) # 580 discoveries

ihw_grenander = IHW(weight_learner = GrenanderLearner(), α = α)
 # 677 discoveries

ihw_grenander_fit1 = fit(ihw_grenander, Ps, Xs, kf)
sum(adjust(ihw_grenander_fit1) .<= α)

Random.seed!(1)
ihw_grenander_fit2 = fit(ihw_grenander, Ps, Xs)
sum(adjust(ihw_grenander_fit2) .<= α)
