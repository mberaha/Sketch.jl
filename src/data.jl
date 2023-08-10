using AdaptiveRejectionSampling
using Distributions
using Random
 
include("utils.jl")


function sample_py_stickbreak(theta, alpha, ndata)
    L = 1000000
    nus = rand.(Beta.(1 - alpha, theta .+ alpha .* collect(1:L)))
    ws = zeros(L)
    ws[1] = nus[1]
    ws[2:(end-1)] = nus[2:(end-1)] .* cumprod(1 .- nus[1:end-2])
    ws[end] = 1.0 - sum(ws[1:end-1])
    
    out = rand(Categorical(ws), (ndata))
    _, idx = unique_ids(out)
    return idx
end

function sample_py_urn(theta, alpha, ndata)
    cnts = [1]
    data = zeros(Core.Int64, ndata)
    data[1] = 1
    k = 1

    for n in 2:ndata
        probas = push!(cnts .- alpha, theta + k * alpha)
        probas = probas ./ sum(probas)
        pos = rand(Categorical(probas))
        data[n] = pos
        if pos == k+1
            cnts = push!(cnts, 1)
            k = k+1
        else
            cnts[pos] += 1
        end
    end

    return data
end

sample_py = sample_py_urn


function sample_u(n, k, p, nsamps=1)
    alpha = p.alpha
    theta = p.theta
    tau = p.tau
                        
    logf_(v) = n * v - (n - k * alpha) * logsumexp([v, log(tau)]) - theta / alpha * ( (tau + exp(v) )^alpha )
    x = range(-10.0 + 0.6 * log(n), log(n), length=200)
    lfmax, maxind = findmax(logf_.(x))                            
    logf(u) = logf_(u) - lfmax

    # run sampler
    delta = -1
    if (n < 5000)
        delta = 0.01
    else
        delta = 0.005
    end
    support = (-Inf, Inf)
    search = (x[maxind] - 5.0, x[maxind] + 5.0)
    sampler = RejectionSampler(logf, support, delta, max_segments=30, logdensity=true, search_range=search)                    
    return exp.(run_sampler!(sampler, nsamps))
end


function sample_ngg(nmax::Int, p::NGGParams, verbose=true)
    cnts = [1, 1]
    k = 2
    for n in 2:(nmax-1)
        if ((n % 100) & verbose) == 0
            println("n: ", n, ", k: ", k)
        end
        
        u = sample_u(n, k, p)[1]
        probas = zeros(k+1)
        probas[end] = p.theta * (u + p.tau)^p.alpha
        probas[1:(end-1)] = (cnts .- p.alpha)
        probas = probas ./ sum(probas)
        
        pos = rand(Categorical(probas))
        if pos == k+1
            cnts = push!(cnts, 1)
            k = k+1
        else
            cnts[pos] += 1
        end
    end
    return cnts
end


struct Zipf
    alpha::Float64
end


function rand_zipf(dist::Zipf, n::Int)
    function random_zipf(a)
        am1 = a - 1.0
        b = 2.0^am1
        
        while true
            U = 1.0 - rand()
            V = rand()
            X = floor(U^(-1.0 / am1))
            
            if X > typemax(Int) || X < 1.0
                continue
            end
            
            T = (1.0 + 1.0 / X)^am1
            
            if V * X * (T - 1.0) / (b - 1.0) <= T / b
                return trunc(X)
            end
        end
    end

    out = [Int(random_zipf(dist.alpha)) for _ in 1:n]
    _, idx = unique_ids(out)
    return idx
    
end
