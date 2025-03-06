using Random, LinearAlgebra, SparseArrays, MAT, Statistics
using DataStructures
using NearestNeighbors

include("structs.jl")

export SuperSparse3Tensor,
       cosine_angle,
       area_triangle,
       mean_square_distance,
       mean_distance_squared,
       max_square_distance,
       distance_matrix,
       time_distance_matrix,
       rbf_similarity_weights_tensor,
       rbf_similarity_weights


###### Weight functions for similarity tensor
function cosine_angle(a,b,c)
    dif1 = (a - b)
    dif2 = (c - b)
    den = (norm(dif1)*norm(dif2))
    if den > 1e-20 cosine = dot(dif1, dif2) / den else cosine = 0 end
    return 1-cosine
end

function area_triangle(a,b,c)
    p = (a+b+c)/2
    return sqrt(p*(p-a)*(p-b)*(p-c))
end

function mean_square_distance(a,b,c)
    p = (a^2+b^2+c^2)/3
    return p
end

function mean_distance_squared(a,b,c)
    p = ((a+b+c)/3)^2
    return p
end

function max_square_distance(a,b,c)
    p = max(a,b,c) ^ 2
    return p
end



######## KNN matrix from dataset
function distance_matrix(X, kn; mode="distance")
    # Convert X to matrix if it's not already
    X_matrix = Matrix(X')  # NearestNeighbors expects points as columns
    
    # Create tree for efficient nearest neighbor search
    tree = BallTree(X_matrix)
    
    # Find k nearest neighbors for each point
    idxs, dists = knn(tree, X_matrix, kn + 1, true)  # k+1 because it includes the point itself
    
    # Create sparse matrix
    n = size(X, 1)
    I = Int[]
    J = Int[]
    V = Float64[]
    
    for i in 1:n
        for (j, d) in zip(idxs[i][2:end], dists[i][2:end])  # Skip first neighbor (self)
            if mode == "distance"
                push!(I, i)
                push!(J, j)
                push!(V, d)
            else  # mode == "connectivity"
                push!(I, i)
                push!(J, j)
                push!(V, 1.0)
            end
        end
    end
    
    K = sparse(I, J, V, n, n)
    K = max.(K, K')
    return K
end


function time_distance_matrix(X,kn; mode="distance")
    print("distance matrix:\t")
    @time distance_matrix(X,kn,mode=mode)
end


function rbf_similarity_weights(T::SuperSparse3Tensor; fast = true)
    valsT = T.V
    valsTT = copy(valsT)
    if fast
        valsTT = exp.(- ((valsT.^2) ./ 4) )
    else

        σ = zeros(Float64,T.n)
        for i in 1:T.n
            σ[i] = maximum(valsT[T.I .== i])
        end

        for (h , (i,j,k,v)) in enumerate(zip(T.I,T.J,T.K,T.V))
            valsTT[h] = exp(-4*v^2 / min(σ[i],σ[j],σ[k])^2 )
        end

    end
    return SuperSparse3Tensor(T.I,T.J,T.K,valsTT,T.n)
end


function rbf_similarity_weights(K::SparseArrays.SparseMatrixCSC; fast=true)
    I, J, valsK = findnz(K)
    n = size(K,1)
    furthest = maximum(K, dims=1)

    if fast
        # W_ij = - ||x_i -x_j||^2 / 2σ^2, for σ = √2
        valsTT = exp.(- ((valsK.^2 ) ./ 4) )
        W = sparse(I,J,valsTT,n,n)
    else
        # S_ij = exp(- 4 ||x_i -x_j||^2 / σ^2), for σ = distance xi to its k-th neighbor
        # W_ij = max(S_ij,S_ji)
        W = spzeros(n,n)
        for (i,j,v) in zip(I,J,valsK)
            W[i,j] = exp(-4*v^2 / min(furthest[i],furthest[j])^2 )
        end
    end
    return W
end
