using Distributions
using SpecialFunctions
using Random

include("params.jl")


function unique_ids(itr)
    v = Vector{eltype(itr)}()
    d = Dict{eltype(itr), Int}()
    revid = Vector{Int}()
    for val in itr
      if haskey(d, val)
        push!(revid, d[val])
      else
        push!(v, val)
        d[val] = length(v)
        push!(revid, length(v))
      end
    end
    (v, revid)
end


function murmur_hash(key::String, seed::UInt32=202303, signed=false)::Int64
    # Implements 32-bit Murmur3 hash.

    function fmix(h::UInt32)::UInt32
        h = h ⊻ (h >> 16)
        h = (h * 0x85ebca6b) & 0xFFFFFFFF
        h = h ⊻ (h >> 13)
        h = (h * 0xc2b2ae35) & 0xFFFFFFFF
        h = h ⊻ (h >> 16)
        return h
    end

    keylen = UInt32(length(key))
    nblocks = UInt32(keylen >> 2)

    h1 = UInt32(seed)

    c1 = 0xcc9e2d51
    c2 = 0x1b873593

    # body
    for block_start in 1:4:nblocks << 2
        # big endian
        k1 = (UInt32(key[block_start + 3]) << 24) |
             (UInt32(key[block_start + 2]) << 16) |
             (UInt32(key[block_start + 1]) << 8) |
             UInt32(key[block_start])

        k1 = (c1 * k1) & 0xFFFFFFFF
        k1 = (k1 << 15 | k1 >> 17) & 0xFFFFFFFF # inlined ROTL32
        k1 = (c2 * k1) & 0xFFFFFFFF

        h1 = h1 ⊻ k1
        h1 = (h1 << 13 | h1 >> 19) & 0xFFFFFFFF # inlined ROTL32
        h1 = (h1 * 5 + 0xe6546b64) & 0xFFFFFFFF
    end

    # tail
    tail_index = nblocks << 2
    k1 = 0
    tail_size = keylen & 3

    if tail_size >= 3
        k1 = k1 ⊻ UInt32(key[tail_index + 2]) << 16
    end
    if tail_size >= 2
        k1 = k1 ⊻ UInt32(key[tail_index + 1]) << 8
    end
    if tail_size >= 1
        k1 = k1 ⊻ UInt32(key[tail_index + 1])
    end

    if tail_size > 0
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = (k1 << 15 | k1 >> 17) & 0xFFFFFFFF # inlined ROTL32
        k1 = (k1 * c2) & 0xFFFFFFFF
        h1 = h1 ⊻ k1
    end

    # finalization
    tmp = UInt32(h1 ⊻ keylen)
    unsigned_val = fmix(tmp)
    if signed
        if unsigned_val & 0x80000000 == 0
            return Int64(unsigned_val)
        else
            return Int64(-(unsigned_val ⊻ 0xFFFFFFFF) + 1)
        end
    else
        return Int64(unsigned_val)
    end
end

function generate_hash(width, seed=202303)
    hash(x) = murmur_hash(string(x), UInt32(seed)) % width + 1

    return hash
end


function hash_dataset(data, hash_fns, width)
    m = length(hash_fns)
    out = zeros((m, width))
    for i = 1:m
        cmap = countmap(hash_fns[i].(data))
        for (k, v) in cmap
            out[i, k] += v
        end
    end

    return out
end


function lbinom(a, b)
    return loggamma(a + 1) - loggamma(a - b) - loggamma(b + 1)
end


function lgammainc(x, a)
    return loggamma(a) + log(gamma_inc(a, x)[1])
end
