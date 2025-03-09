include("src/utils.jl")
include("src/tensors.jl")
include("src/CV_helpers.jl")
include("src/functions.jl")
include("src/labelspreading.jl")
include("src/error_distributions.jl")

using YAML
using DataFrames
using CSV

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
function run_analysis(mode, data, y, features, A, DG_isqrt, T=nothing, DH_isqrt=nothing, B=nothing, known_indices=nothing; use_perturbed=false, σ=0.2)
    # Create a copy of the data dictionary to avoid modifying the original
    data_copy = deepcopy(data)
    
    # Get configuration parameters
    balanced, binary, alphas, betas, weight_function, mixing_functions, ε, kn, noise, percentage_of_known_labels, num_trials, dataset_name, data_type = prepare_config_data(data_copy[mode])
    
    # Create one-hot encoded matrix for all labels
    n = length(y)
    num_classes = length(unique(y))
    Y = zeros(Float64, n, num_classes)
    for (i, label) in enumerate(y)
        Y[i, label] = 1.0
    end
    
    # Initialize matrix for known labels (will be probability distributions)
    known_labels = zeros(Float64, n, num_classes)
    
    # Debug print for number of known labels
    println("Number of known labels: ", length(known_indices), " out of ", n, " total labels")
    
    # Process the known labels
    for i in known_indices
        if use_perturbed
            # For perturbed case, create probability distribution for known labels
            _, Ỹ = create_error_distribution(Y[i:i, :], σ)
            known_labels[i, :] = Ỹ[1, :]
        else
            # For unperturbed case, use original one-hot encoded labels
            known_labels[i, :] = Y[i, :]
        end
    end
    
    # Create a mask for unknown labels (for accuracy calculation)
    unknown_indices = setdiff(1:n, known_indices)
    
    if mode == "HOLS"
        # Initialize results array
        results = []
        
        # Run label spreading for each class
        X_labels = zeros(length(mixing_functions), n, num_classes)
        
        for (j, f) in enumerate(mixing_functions)
            # Initialize predictions matrix for all classes
            X_all = zeros(Float64, n, num_classes)
            
            for class in 1:num_classes
                # Get the known labels for this class
                Y_class = zeros(Float64, n)
                for i in known_indices
                    Y_class[i] = known_labels[i, class]
                end
                
                # Add small epsilon to avoid numerical issues
                Y_class = (1 - ε) .* Y_class .+ ε
                
                # Normalize if needed
                tildeY = copy(Y_class)
                if f == f_max
                    # For f_max, we don't need to normalize
                    X_class, _ = projected_second_order_label_spreading(
                        x -> Tf(T, DH_isqrt, f, x),
                        x -> Ax(A, DG_isqrt, x),
                        tildeY,
                        alphas[1], betas[1], 1-alphas[1]-betas[1],
                        x -> maximum(abs.(x)))
                else
                    if φ(DH_isqrt .* tildeY, f, B) > 1e-20
                        tildeY = tildeY ./ φ(tildeY, f, B)
                    end
                    X_class, _ = projected_second_order_label_spreading(
                        x -> Tf(T, DH_isqrt, f, x),
                        x -> Ax(A, DG_isqrt, x),
                        tildeY,
                        alphas[1], betas[1], 1-alphas[1]-betas[1],
                        x -> φ(DH_isqrt .* x, f, B))
                end
                
                # Store predictions for this class
                X_all[:, class] = X_class
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
            result = (j, dataset_name, n, kn, alphas[1]+betas[1], alphas[1], betas[1], 
                     percentage_of_known_labels[1]*100, balanced, acc, prec, rec)
            push!(results, result)
        end
        
        data = results
    end
    
    colnames = [:method_name, :dataset, :size, :knn, :alpha_plus_beta, :alpha, :beta, 
                :percentage_known, :balanced, :accuracy, :precision, :recall]
    
    # Convert data to DataFrame with explicit types
    df = DataFrame(
        method_name = [x[1] for x in data],
        dataset = [string(x[2]) for x in data],
        size = [Int(x[3]) for x in data],
        knn = [Int(x[4]) for x in data],
        alpha_plus_beta = [Float64(x[5]) for x in data],
        alpha = [Float64(x[6]) for x in data],
        beta = [Float64(x[7]) for x in data],
        percentage_known = [Float64(x[8]) for x in data],
        balanced = [Bool(x[9]) for x in data],
        accuracy = [Float64(x[10]) for x in data],
        precision = [Float64(x[11]) for x in data],
        recall = [Float64(x[12]) for x in data]
    )
    
    return df
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

    # Load configuration
    data = YAML.load(open("config.yml"))
    
    # Ensure we're in HOLS mode
    data["mode"] = "HOLS"
    mode = "HOLS"
    
    # Load data
    balanced, binary, alphas, betas, weight_function, mixing_functions, ε, kn, noise, _, num_trials, dataset_name, data_type = prepare_config_data(data[mode])
    features, y, A, DG_isqrt, T, DH_isqrt, B = load_data(dataset_name, kn, noise, weight_function, mode, binary)
    
    # Set parameters for the analysis
    σ_values = [0.25,0.5,0.75,1.0]  #collect(0.0:0.05:1.5)  # Standard deviation values from 0.0 to 1.0 in steps of 0.05
    percentages = [5,10,15,20.0]  # Percentage of known labels
    num_trials = 10  # Number of trials to average over
    
    for σ in σ_values
        println("\nProcessing with perturbation σ = $σ")
        for pct in percentages
            println("Processing $(pct)% known labels...")
            
            # Update configuration
            data["HOLS"]["percentage_of_known_labels"] = [pct/100]  # Convert to decimal
            
            # Initialize arrays to store results for averaging
            trial_results = []
            
            # Run multiple trials
            for trial in 1:num_trials
                println("\nTrial $trial of $num_trials")
                
                # Select known labels for this trial
                known_indices = select_known_labels(y, pct/100, balanced)
                
                # Run unperturbed analysis
                unperturbed_df = run_analysis(mode, data, y, features, A, DG_isqrt, T, DH_isqrt, B, known_indices, use_perturbed=false)
                
                # Run perturbed analysis with same known labels
                perturbed_df = run_analysis(mode, data, y, features, A, DG_isqrt, T, DH_isqrt, B, known_indices, use_perturbed=true, σ=σ)
                
                # Store results for this trial
                for (u_row, p_row) in zip(eachrow(unperturbed_df), eachrow(perturbed_df))
                    push!(trial_results, (
                        model_used = string(u_row.method_name),
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
                end
            end
            
            # Group results by model and compute averages
            grouped_results = Dict()
            for result in trial_results
                key = result.model_used
                if !haskey(grouped_results, key)
                    grouped_results[key] = []
                end
                push!(grouped_results[key], result)
            end
            
            # Compute and store averages
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
        end
    end
    
    # Save results
    CSV.write("perturbed_results/perturbation_analysis_averaged.csv", results)
    println("Results saved to perturbed_results/perturbation_analysis_averaged.csv")
end

# Run the analysis
run_perturbation_analysis() 