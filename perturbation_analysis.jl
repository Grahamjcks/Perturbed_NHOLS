include("src/utils.jl")
include("src/tensors.jl")
include("src/CV_helpers.jl")
include("src/functions.jl")
include("src/labelspreading.jl")
include("src/error_distributions.jl")

using YAML
using DataFrames
using CSV
using LinearAlgebra
using DelimitedFiles  # For readdlm

# Create results directory if it doesn't exist
if !isdir("perturbed_results")
    mkdir("perturbed_results")
end

# Function to select known labels for each class
function select_known_labels(y, percentage, balanced)
    n = length(y)
    num_classes = length(unique(y))
    
    # Calculate number of labels per class
    if balanced
        # For balanced case, use equal number for each class
        labels_per_class = Int[]
        for i in 1:num_classes
            class_size = length(findall(==(i), y))
            num_labels = max(1, Int(round(percentage * class_size)))  # At least 1 label per class
            push!(labels_per_class, num_labels)
        end
    else
        # For unbalanced case, use percentage of each class
        labels_per_class = Int[]
        for i in 1:num_classes
            class_size = length(findall(==(i), y))
            num_labels = max(1, Int(round(percentage * class_size)))  # At least 1 label per class
            push!(labels_per_class, num_labels)
        end
    end
    
    # Debug print for label distribution
    println("Labels per class: ", labels_per_class)
    println("Total labels to select: ", sum(labels_per_class))
    
    known_indices = Int[]
    for (label, num_known) in enumerate(labels_per_class)
        class_indices = findall(y .== label)
        if !isempty(class_indices)  # Only process if we have samples for this class
            shuffle!(class_indices)
            num_known = min(num_known, length(class_indices))  # Don't try to take more than we have
            known_class_indices = class_indices[1:num_known]
            append!(known_indices, known_class_indices)
        end
    end
    
    return sort(known_indices)  # Sort to ensure consistent ordering
end

# Function to run analysis with either original or perturbed labels
function run_analysis(mode, data, y, features, A, DG_isqrt, T, DH_isqrt, B, known_indices; use_perturbed=false, σ=0.0)
    # Get parameters from config
    if mode == "HOLS"
        balanced = data[mode]["balanced"]
        binary = data[mode]["binary"]
        alphas = data[mode]["α"]
        betas = data[mode]["β"]
        weight_function = data[mode]["weight_function"]
        mixing_functions = data[mode]["mixing_functions"]
        ε = data[mode]["ε"]
        kn = data[mode]["kn"]
        noise = data[mode]["noise"]
        percentage_of_known_labels = data[mode]["percentage_of_known_labels"]
        dataset_name = data[mode]["dataset"]
        data_type = data[mode]["data_type"]
    else # LS mode
        balanced = data[mode]["balanced"]
        binary = data[mode]["binary"]
        alphas = data[mode]["α"]
        ε = data[mode]["ε"]
        kn = data[mode]["kn"]
        noise = data[mode]["noise"]
        percentage_of_known_labels = data[mode]["percentage_of_known_labels"]
        dataset_name = data[mode]["dataset"]
        data_type = data[mode]["data_type"]
        betas = [1.0 - alphas[1]]  # In LS, beta = 1 - alpha
    end
    
    # Get dataset size
    n = size(features, 1)
    
    # Get number of classes
    num_classes = length(unique(y))
    
    # Initialize results array
    results = []
    
    # Initialize matrices to store convergent solutions
    X_all_unperturbed = zeros(Float64, n, num_classes)
    X_all_perturbed = zeros(Float64, n, num_classes)
    
    # Initialize error matrix Z
    Z = zeros(Float64, n, num_classes)
    
    # Get known and unknown indices
    unknown_indices = setdiff(1:n, known_indices)
    
    # Create one-hot encoding for known labels
    known_labels = zeros(Float64, n, num_classes)
    for i in known_indices
        known_labels[i, y[i]] = 1.0
    end
    
    # Add perturbation if requested
    tildeY = copy(known_labels)
    if use_perturbed && σ > 0
        # Generate random perturbation matrix
        perturbation = randn(n, num_classes) .* σ
        
        # Only perturb known labels
        for i in known_indices
            for j in 1:num_classes
                tildeY[i, j] += perturbation[i, j]
            end
        end
        
        # Store the error matrix Z
        Z = perturbation
    end
    
    # Initialize predictions matrix for all classes
    X_all = zeros(Float64, n, num_classes)
    
    # Define common functions used by both modes
    Ax(A, DG_isqrt, x) = DG_isqrt * (A * (DG_isqrt * x))
    
    # Define functions for HOLS
    if mode == "HOLS"
        # Define mixing function
        if mixing_functions == "f_max"
            f(x) = maximum(abs.(x))
        else
            error("Mixing function $mixing_functions not supported")
        end
        
        # Define functions for HOLS
        Tf(T, DH_isqrt, f, x) = T * (DH_isqrt * (x ./ f(x)))
        φ(x, f, B) = f(B * x)
        
        # Process each class separately
        for class in 1:num_classes
            # Get the known labels for this class
            Y_class = zeros(Float64, n)
            for i in known_indices
                Y_class[i] = tildeY[i, class]
            end
            
            # Add small epsilon to avoid numerical issues
            Y_class = (1 - ε) .* Y_class .+ ε
            
            # Use projected second order label spreading for HOLS mode
            if f == (x -> maximum(abs.(x)))
                X_class, _, _, _ = projected_second_order_label_spreading(
                    x -> Tf(T, DH_isqrt, f, x),
                    x -> Ax(A, DG_isqrt, x),
                    Y_class,
                    alphas[1], betas[1], 1-alphas[1]-betas[1],
                    x -> maximum(abs.(x)))
            else
                if φ(DH_isqrt .* Y_class, f, B) > 1e-20
                    Y_class = Y_class ./ φ(Y_class, f, B)
                end
                X_class, _, _, _ = projected_second_order_label_spreading(
                    x -> Tf(T, DH_isqrt, f, x),
                    x -> Ax(A, DG_isqrt, x),
                    Y_class,
                    alphas[1], betas[1], 1-alphas[1]-betas[1],
                    x -> φ(DH_isqrt .* x, f, B))
            end
            
            # Store predictions for this class
            X_all[:, class] = X_class
            
            # Store the convergent solution for unperturbed or perturbed case
            if use_perturbed
                X_all_perturbed[:, class] = X_class
            else
                X_all_unperturbed[:, class] = X_class
            end
        end
    else # LS mode
        # Initialize results array
        results = []
        
        for class in 1:num_classes
            # Get the known labels for this class
            Y_class = zeros(Float64, n)
            for i in known_indices
                Y_class[i] = tildeY[i, class]
            end
            
            # Add small epsilon to avoid numerical issues
            Y_class = (1 - ε) .* Y_class .+ ε
            
            # Use standard label spreading for LS mode
            X_class, _, _ = standard_label_spreading(
                x -> Ax(A, DG_isqrt, x),
                Y_class,
                alphas[1], 1-alphas[1])
            
            # Store predictions for this class
            X_all[:, class] = X_class
            
            # Store the convergent solution for unperturbed or perturbed case
            if use_perturbed
                X_all_perturbed[:, class] = X_class
            else
                X_all_unperturbed[:, class] = X_class
            end
        end
    end
    
    # Normalize predictions to ensure they sum to 1
    row_sums = sum(X_all, dims=2)
    X_all ./= row_sums
    
    # Calculate accuracy on unknown labels only
    Y_pred = zeros(Int, length(unknown_indices))
    for i in 1:length(unknown_indices)
        # Get predicted probabilities for this node
        probs = X_all[unknown_indices[i], :]
        # Find the class with maximum probability
        _, pred_class = findmax(probs)
        Y_pred[i] = pred_class
    end
    Y_real = y[unknown_indices]
    
    # Debug output
    println("Known labels distribution: ", [count(==(c), y[known_indices]) for c in 1:num_classes])
    println("Unknown labels distribution: ", [count(==(c), Y_real) for c in 1:num_classes])
    println("Predicted labels distribution: ", [count(==(c), Y_pred) for c in 1:num_classes])
    
    acc = accuracy(Y_pred, Y_real)
    prec = precision(Y_pred, Y_real)
    rec = recall(Y_pred, Y_real)
    
    # Create result tuple
    result = (mode == "HOLS" ? "HOLS" : "LS", dataset_name, n, kn, alphas[1]+betas[1], alphas[1], betas[1], 
             percentage_of_known_labels[1]*100, balanced, acc, prec, rec)
    push!(results, result)
    
    colnames = [:method_name, :dataset, :size, :knn, :alpha_plus_beta, :alpha, :beta, 
                :percentage_known, :balanced, :accuracy, :precision, :recall]
    
    # Convert data to DataFrame with explicit types
    df = DataFrame(
        method_name = [x[1] for x in results],
        dataset = [string(x[2]) for x in results],
        size = [Int(x[3]) for x in results],
        knn = [Int(x[4]) for x in results],
        alpha_plus_beta = [Float64(x[5]) for x in results],
        alpha = [Float64(x[6]) for x in results],
        beta = [Float64(x[7]) for x in results],
        percentage_known = [Float64(x[8]) for x in results],
        balanced = [Bool(x[9]) for x in results],
        accuracy = [Float64(x[10]) for x in results],
        precision = [Float64(x[11]) for x in results],
        recall = [Float64(x[12]) for x in results]
    )
    
    return df, X_all_unperturbed, X_all_perturbed, Z, tildeY
end

# Function to load data for LS algorithm (simpler version without T, DH_isqrt, B)
function load_data_simple(dataset_name, kn, noise, mode, binary)
    # This is a simplified version of load_data for LS mode
    # It only returns features, y, A, and DG_isqrt
    
    # Load the dataset
    if mode == "LS"
        if dataset_name == "Rice31" || dataset_name == "Caltech36"
            # Load fb100 data
            if dataset_name == "Rice31"
                edges_file = "data/fb100/edges-Rice31.txt"
                labels_file = "data/fb100/labels-Rice31.txt"
            else # Caltech36
                edges_file = "data/fb100/edges-Caltech36.txt"
                labels_file = "data/fb100/labels-Caltech36.txt"
            end
            
            # Read edges and labels
            edges = readdlm(edges_file, Int)
            labels = readdlm(labels_file, Int)
            
            # Create graph
            n = maximum(edges)
            A = sparse(edges[:, 1], edges[:, 2], 1, n, n)
            A = A + A'
            
            # Create features (not used in this algorithm)
            features = zeros(n, 1)
            
            # Get labels - handle both single column and multi-column cases
            if size(labels, 2) > 1
                y = labels[:, 2]  # If labels has multiple columns, use the second column
            else
                y = labels[:, 1]  # If labels has only one column, use the first column
            end
            
            # Calculate DG_isqrt
            D = Diagonal(vec(sum(A, dims=2)))
            DG_isqrt = Diagonal(1 ./ sqrt.(diag(D)))
            
            return features, y, A, DG_isqrt
        else
            error("Dataset $dataset_name not supported for LS mode")
        end
    else
        error("Mode $mode not supported for load_data_simple")
    end
end

# Main function to run perturbation analysis
function run_perturbation_analysis()
    # Initialize results DataFrame with specified column types
    results = DataFrame(
        model_used = String[],
        perturbed_standard_deviation = Float64[],
        data_size = String[],
        knn = Int64[],
        percentage_of_known_labels = Float64[],
        α = Float64[],
        β = Float64[],
        unperturbed_accuracy = Float64[],
        perturbed_accuracy = Float64[],
        perturbed_difference = Float64[]
    )
    
    # Initialize error bounds DataFrame with the exact column names requested
    error_bounds = DataFrame(
        convergent_solution_L2_difference = Float64[],
        error_matrix_L2 = Float64[],
        percentage_of_known_labels = Float64[],
        model_used = String[],
        perturbed_standard_deviation = Float64[],
        data_size = Int64[],
        knn = Int64[],
        α = Float64[],
        β = Float64[]
    )

    # Load configuration
    data = YAML.load(open("config.yml"))
    
    # Get the mode from config
    mode = data["mode"]
    
    # Load data based on mode
    if mode == "HOLS"
        balanced, binary, alphas, betas, weight_function, mixing_functions, ε, kn, noise, percentage_of_known_labels, num_trials, dataset_name, data_type = prepare_config_data(data[mode])
        features, y, A, DG_isqrt, T, DH_isqrt, B = load_data(dataset_name, kn, noise, weight_function, mode, binary)
    else # LS mode
        balanced = data[mode]["balanced"]
        binary = data[mode]["binary"]
        alphas = data[mode]["α"]
        ε = data[mode]["ε"]
        kn = data[mode]["kn"]
        noise = data[mode]["noise"]
        percentage_of_known_labels = data[mode]["percentage_of_known_labels"]
        num_trials = data[mode]["num_trials"]
        dataset_name = data[mode]["dataset"]
        data_type = data[mode]["data_type"]
        
        # Load data for LS mode (simpler, doesn't need T, DH_isqrt, B)
        features, y, A, DG_isqrt = load_data_simple(dataset_name, kn, noise, mode, binary)
        T = nothing
        DH_isqrt = nothing
        B = nothing
    end
    
    # Set parameters for the analysis
    σ_values = collect(0.0:0.05:1.0)  # Standard deviation values from 0.0 to 1.0 in steps of 0.05
    
    # Convert percentage_of_known_labels from decimal to percentage
    percentages = [p * 100 for p in percentage_of_known_labels]
    
    # Use the number of trials from the config
    num_trials_to_run = num_trials
    
    for σ in σ_values
        println("\nProcessing with perturbation σ = $σ")
        for pct in percentages
            println("Processing $(pct)% known labels...")
            
            # Update configuration
            data[mode]["percentage_of_known_labels"] = [pct/100]  # Convert to decimal
            
            # Initialize arrays to store results for averaging
            trial_results = []
            trial_error_bounds = []
            
            # Run multiple trials
            for trial in 1:num_trials_to_run
                println("\nTrial $trial of $num_trials_to_run")
                
                # Select known labels for this trial
                known_indices = select_known_labels(y, pct/100, balanced)
                
                # Run unperturbed analysis
                unperturbed_df, X_unperturbed, _, _, _ = run_analysis(mode, data, y, features, A, DG_isqrt, T, DH_isqrt, B, known_indices, use_perturbed=false)
                
                # Run perturbed analysis with same known labels
                perturbed_df, _, X_perturbed, Z, _ = run_analysis(mode, data, y, features, A, DG_isqrt, T, DH_isqrt, B, known_indices, use_perturbed=true, σ=σ)
                
                # Calculate L2 norms
                solution_diff_norm = norm(X_unperturbed - X_perturbed)
                error_matrix_norm = norm(Z)
                
                # Store results for this trial
                for (u_row, p_row) in zip(eachrow(unperturbed_df), eachrow(perturbed_df))
                    push!(trial_results, (
                        model_used = mode == "HOLS" ? "HOLS" : "LS",
                        perturbed_standard_deviation = Float64(σ),
                        data_size = string(u_row.size),
                        knn = Int(u_row.knn),
                        percentage_of_known_labels = Float64(pct),
                        α = Float64(u_row.alpha),
                        β = Float64(u_row.beta),
                        unperturbed_accuracy = Float64(u_row.accuracy),
                        perturbed_accuracy = Float64(p_row.accuracy),
                        perturbed_difference = Float64(p_row.accuracy - u_row.accuracy)
                    ))
                    
                    push!(trial_error_bounds, (
                        convergent_solution_L2_difference = Float64(solution_diff_norm),
                        error_matrix_L2 = Float64(error_matrix_norm),
                        percentage_of_known_labels = Float64(pct),
                        model_used = mode == "HOLS" ? "HOLS" : "LS",
                        perturbed_standard_deviation = Float64(σ),
                        data_size = Int(u_row.size),
                        knn = Int(u_row.knn),
                        α = Float64(u_row.alpha),
                        β = Float64(u_row.beta)
                    ))
                end
            end
            
            # Group results by model and compute averages
            grouped_results = Dict()
            grouped_error_bounds = Dict()
            
            for result in trial_results
                key = result.model_used
                if !haskey(grouped_results, key)
                    grouped_results[key] = []
                end
                push!(grouped_results[key], result)
            end
            
            for result in trial_error_bounds
                key = result.model_used
                if !haskey(grouped_error_bounds, key)
                    grouped_error_bounds[key] = []
                end
                push!(grouped_error_bounds[key], result)
            end
            
            # Compute and store averages for perturbation results
            for (model, model_results) in grouped_results
                avg_result = (
                    model_used = model,
                    perturbed_standard_deviation = σ,
                    data_size = model_results[1].data_size,
                    knn = model_results[1].knn,
                    percentage_of_known_labels = pct,
                    α = model_results[1].α,
                    β = model_results[1].β,
                    unperturbed_accuracy = mean([r.unperturbed_accuracy for r in model_results]),
                    perturbed_accuracy = mean([r.perturbed_accuracy for r in model_results]),
                    perturbed_difference = mean([r.perturbed_difference for r in model_results])
                )
                push!(results, avg_result)
            end
            
            # Compute and store averages for error bounds
            for (model, model_results) in grouped_error_bounds
                avg_result = (
                    convergent_solution_L2_difference = mean([r.convergent_solution_L2_difference for r in model_results]),
                    error_matrix_L2 = mean([r.error_matrix_L2 for r in model_results]),
                    percentage_of_known_labels = pct,
                    model_used = model,
                    perturbed_standard_deviation = σ,
                    data_size = model_results[1].data_size,
                    knn = model_results[1].knn,
                    α = model_results[1].α,
                    β = model_results[1].β
                )
                push!(error_bounds, avg_result)
            end
        end
    end
    
    # Save results
    CSV.write("perturbed_results/perturbation_analysis_averaged.csv", results)
    CSV.write("perturbed_results/error_bounds.csv", error_bounds)
    println("Results saved to perturbed_results/perturbation_analysis_averaged.csv")
    println("Error bounds saved to perturbed_results/error_bounds.csv")
end

# Run the analysis
run_perturbation_analysis() 