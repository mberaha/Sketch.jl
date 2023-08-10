using Optimization
using OptimizationOptimJL
using SpecialFunctions
using Integrals
import NNlib.logsumexp

include("params.jl")

function log_marg_dp(theta, sketch::AbstractVector, n::Integer)
    J = length(sketch)
    out =  loggamma(theta) - loggamma(theta + n)
    out += sum(loggamma.( theta / J .+ sketch))
    out -= J * loggamma(theta / J)
    return out
end

function fit_dp(sketch)
    n = Int(sum(sketch))
    theta0 = 100.0

    obj(x, p) = -log_marg_dp(x[1], sketch, n)
    f = OptimizationFunction(obj, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(f, [theta0], lb=[0.0], ub=[Inf]) 
    tmp = solve(prob, BFGS())
    out = DPParams(tmp.u[1])
    return out
end

function log_bessel_halfint(z, nu)
    if (2 * nu) % 1 != 0
         throw(DomainError(nu, "Expected argument nu to be half integer"))
    end

    jrange = 0:Int(nu - 0.5)
    out = 0.5 * log(pi / 2) - z - 0.5 * log(z)
    out += logsumexp(
        loggamma.(jrange .+ nu .+ 0.5) .- loggamma.(jrange .+ 1) .- loggamma.(nu .- jrange .+ 0.5) .- jrange * log(2 * z)
    )
    return out
end

function log_marg_nig(theta, sketch::AbstractVector, n::Integer)
    J = length(sketch)

    function log_integrand(x)
        return (n-1) * log(x) - (n/2 - J/4) * log(1 + 2*x) + sum(
            log_bessel_halfint.(theta / J * sqrt(1 + 2 * x) , abs.(sketch .- 0.5)))
    end

    function eval_log_integral()
        # Evaluates the integral using the log sum exp trick
        obj(x, u) =  -log_integrand(x[1])
        f = OptimizationFunction(obj, Optimization.AutoForwardDiff())
        prob = OptimizationProblem(f, [10.0], lb=[0.000001], ub=[Inf]) 
        tmp = solve(prob, BFGS())
        fmax = log_integrand(tmp.u[1])
        integrand(x, p) = exp(log_integrand(x) - fmax)
        integral = IntegralProblem(
            integrand,  max(tmp.u[1] - 200, 0), tmp.u[1] + 1000, [])
        eval_int = solve(integral, QuadGKJL()).u
        return fmax + log(eval_int)
    end
    
    out = (n + J/2) * log(theta / J) + theta
    out += eval_log_integral()

    return out
end


function fit_nig(sketch)
    n = Int(sum(sketch))
    theta0 = 100.0

    obj(x, p) = -log_marg_nig(x[1], sketch, n)
    f = OptimizationFunction(obj, Optimization.AutoFiniteDiff())
    prob = OptimizationProblem(f, [theta0], lb=[1.0], ub=[Inf]) 
    tmp = solve(prob, Optim.GradientDescent())
    theta_opt = tmp.u[1]
    out = NGGParams(theta_opt, 0.5, 0.5)
    return out
end


function log_marg_ngg(params::NGGParams, cnts)

    n = sum(cnts)
    k = length(cnts)
    theta = params.theta
    alpha = params.alpha
    tau = params.tau


    function log_integrand(u)
        return (n-1) * log(u) - theta / alpha * (tau + u)^alpha + (k * alpha - n) * log(tau + u)
    end

    function eval_log_integral()
        # Evaluates the integral using the log sum exp trick
        obj(x, u) =  -log_integrand(x[1])
        f = OptimizationFunction(obj, Optimization.AutoForwardDiff())
        prob = OptimizationProblem(f, [10.0], lb=[0.000001], ub=[Inf]) 
        tmp = solve(prob, BFGS())
        fmax = log_integrand(tmp.u[1])
        integrand(x, p) = exp(log_integrand(x) - fmax)
        integral = IntegralProblem(
            integrand, max(tmp.u[1] - 1000, 0), tmp.u[1] + 1000, [])
        eval_int = solve(integral, QuadGKJL()).u
        return fmax + log(eval_int)
    end


    out = eval_log_integral()
    out += k * log(theta) - loggamma(n) + theta / alpha * tau^alpha + sum(
        loggamma.(cnts .+ 1.0) .- loggamma(1.0 - alpha))
    return out
end


function log_marg_ngg(theta::Float64, alpha::Float64, tau::Float64, cnts)
    p = NGGParams(theta, alpha, tau)
    return log_marg_ngg(p, cnts)
end


function fit_ngg(data)
    cnts = [x[2] for x in countmap(data)]
    p0 = [[100.0, 0.3],
          [1000.0, 0.3],
          [20000.0, 0.3]]
    tau = 0.5

    function solve_opt(x0)
        obj(x, p) = -Sketch.log_marg_ngg(x[1], x[2], tau, cnts)
        f = OptimizationFunction(obj, Optimization.AutoFiniteDiff())
        prob = OptimizationProblem(f, x0, lb=[0.5, 0.0001], ub=[Inf, 0.9999])
        opt_x = x0
        opt_f = obj(x0, [])
        try 
            tmp = solve(prob, BFGS())
            opt_x = tmp.u
            opt_f = tmp.objective
        catch(e)
            
        end
        return opt_x, opt_f
    end

    res = [solve_opt(x) for x in p0]
    opt_ind = findmin([x[2] for x in res])[2]
    theta, alpha = res[opt_ind][1]
    out = NGGParams(theta, alpha, tau)
    return out
end