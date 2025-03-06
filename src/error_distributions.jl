using Random, Distributions, LinearAlgebra

"""
    create_error_distribution(Y::Matrix{Float64}, σ::Float64=0.1)

Creates an error matrix Z based on a half-normal distribution with standard deviation σ.
For each node i, if j* is the index of the true label (Y[i,j*] = 1):
- Z[i,j*] follows |N(0,σ²)|
- Z[i,ℓ] = -Z[i,j*]/(L-1) for ℓ ≠ j*
where L is the number of classes.

This ensures that Ỹ = Y - Z remains a valid probability distribution (rows sum to 1).

Parameters:
- Y: One-hot encoded label matrix where Y[i,j] = 1 if node i has label j
- σ: Standard deviation for the half-normal distribution (default: 0.1)

Returns:
- Z: Error matrix with the same dimensions as Y
- Ỹ: Modified label matrix (Y - Z)
"""
function create_error_distribution(Y::Matrix{Float64}, σ::Float64=0.1)
    n, L = size(Y)  # n nodes, L classes
    Z = zeros(n, L)
    
    # Create normal distribution and take absolute values to get half-normal
    normal_dist = Normal(0.0, σ)
    
    for i in 1:n
        # Find the true label index (j*)
        j_star = findfirst(x -> x ≈ 1.0, Y[i,:])
        
        if !isnothing(j_star)
            # Sample error for the true label from half-normal (abs of normal)
            Z[i,j_star] = abs(rand(normal_dist))
            
            # Distribute the negative error evenly across other classes
            error_per_class = -Z[i,j_star] / (L - 1)
            for ℓ in 1:L
                if ℓ != j_star
                    Z[i,ℓ] = error_per_class
                end
            end
        end
    end
    
    # Create modified label matrix
    Ỹ = Y - Z
    
    # Verify that rows still sum to 1 (within numerical precision)
    @assert all(abs.(sum(Ỹ, dims=2) .- 1) .< 1e-10) "Modified labels do not sum to 1"
    
    return Z, Ỹ
end

"""
    verify_error_distribution(Z::Matrix{Float64}, Y::Matrix{Float64})

Verifies that the error matrix Z satisfies the conditions from Proposition 1.1:
For each node i with true label j*, Z[i,j*] - ∑_{ℓ≠j*} Z[i,ℓ] = 0

Parameters:
- Z: Error matrix
- Y: Original one-hot encoded label matrix

Returns:
- Boolean indicating whether all conditions are satisfied
"""
function verify_error_distribution(Z::Matrix{Float64}, Y::Matrix{Float64})
    n, L = size(Y)
    for i in 1:n
        j_star = findfirst(x -> x ≈ 1.0, Y[i,:])
        if !isnothing(j_star)
            # Calculate Z[i,j*] - ∑_{ℓ≠j*} Z[i,ℓ]
            error_sum = Z[i,j_star] - sum(Z[i,ℓ] for ℓ in 1:L if ℓ != j_star)
            if abs(error_sum) > 1e-10
                @warn "Condition violated for node $i: error sum = $error_sum"
                return false
            end
        end
    end
    return true
end 