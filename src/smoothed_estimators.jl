using SpecialFunctions
using NNlib
using Optimization
using OptimizationOptimJL
using Random
using Distributions
using StatsBase

include("params.jl")


function freq_est(params::DPParams, Cj::Integer, J::Integer)
    return  Cj * J / (params.theta + J)
end


function card_est(params::DPParams, sketch::AbstractVector)
    n = sum(sketch)
    J = length(sketch)
    qs = sketch ./ n
    theta = params.theta

    out = 0
    for j in 1:J
        out += qs[j] / (1 + sketch[j])  * (
            digamma(sketch[j] + 1 + theta/J) - digamma(theta / J))
    end

    out = n * theta / J * out
    return out 
end


function freq_est(params::NGGParams, Cj::Integer, J::Integer)
    betaj = params.theta / (params.alpha * J) * params.tau^params.alpha

    return Cj * (1 - params.alpha) * (
        1 - betaj * exp(betaj) * expint(1.0 / params.alpha, betaj))
end


function card_est(params::NGGParams, sketch::AbstractVector, n_mc=1000)
    function sample_v()
        es = rand(Exponential(1.0), n_mc)
        bs = rand(Beta(1 - params.alpha, params.alpha), n_mc)
        betaj = params.theta / (params.alpha * J) * params.tau^params.alpha
        vs = bs .* (1.0 .- (betaj ./ (betaj .+ es)).^(1.0 / params.alpha) )
        return vs
    end

    function mc_integral(vs, j)
        cj = sketch[j]
        integrand(v) = (1.0 - (1.0 - v)^(cj - 1)) / v
        return mean(integrand.(vs))
    end
    
    n = sum(sketch)
    J = length(sketch)
    qs = sketch ./ n

    out = 0
    vs = sample_v()
    for j in 1:J
        out += qs[j] / (1 + sketch[j])  * mc_integral(vs, j)
    end
    out = n * out
    return out 
end