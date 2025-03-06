include("src/utils.jl")
include("src/tensors.jl")
include("src/CV_helpers.jl")
include("src/functions.jl")
include("src/labelspreading.jl")
include("src/error_distributions.jl")

using YAML

data = YAML.load(open("config.yml"))
mode = data["mode"]

# Create results directories if they don't exist
if !isdir("./results")
   mkdir("./results")
end
if !isdir("./perturbed_results")
   mkdir("./perturbed_results")
end

colnames = [:method_name, :size, :knn, :percentage_of_known_labels, :α, :β, :balanced, :noise, :acc, :prec, :rec]

# Function to run analysis with either original or perturbed labels
function run_analysis(mode, data, y, features, A, DG_isqrt, T=nothing, DH_isqrt=nothing, B=nothing; use_perturbed=false)
    if use_perturbed
        # Convert y to one-hot encoded matrix
        n = length(y)
        num_classes = length(unique(y))
        Y = zeros(Float64, n, num_classes)
        for (i, label) in enumerate(y)
            Y[i, label] = 1.0
        end
        
        # Create perturbed labels
        Z, Ỹ = create_error_distribution(Y)
        
        # Convert Ỹ back to label format
        y_perturbed = [argmax(Ỹ[i,:]) for i in 1:n]
        
        # Use perturbed labels for analysis
        y = y_perturbed
    end
    
    if mode == "LS"
        balanced, binary, alphas, _, _, _, ε, kn, noise, percentage_of_known_labels, num_trials, dataset_name, data_type = prepare_config_data(data[mode])
        data = analyze_dataset_LS(dataset_name, num_trials,
                                alphas,
                                kn,
                                A, DG_isqrt,
                                percentage_of_known_labels,
                                y,
                                balanced,
                                ε)
    elseif mode == "HOLS"
        balanced, binary, alphas, betas, weight_function, mixing_functions, ε, kn, noise, percentage_of_known_labels, num_trials, dataset_name, data_type = prepare_config_data(data[mode])
        data = analyze_dataset_HOLS(dataset_name, num_trials,
                                  kn,
                                  A, DG_isqrt, T, DH_isqrt, B, φ,
                                  mixing_functions,
                                  percentage_of_known_labels,
                                  y,
                                  balanced,
                                  alphas,
                                  betas,
                                  ε)
    elseif mode == "both"
        balanced, binary, alphas, betas, weight_function, mixing_functions, ε, kn, noise, percentage_of_known_labels, num_trials, dataset_name, data_type = prepare_config_data(data[mode])
        data = analyze_dataset(dataset_name, num_trials,
                             kn,
                             A, DG_isqrt, T, DH_isqrt, B, φ,
                             mixing_functions,
                             percentage_of_known_labels,
                             y,
                             balanced,
                             alphas,
                             betas,
                             ε)
    end
    
    df = DataFrame(data)
    println("DataFrame columns before rename: ", names(df))
    println("Attempting to rename to: ", colnames)
    rename!(df, colnames)
    
    # Save results in appropriate directory
    output_dir = use_perturbed ? "./perturbed_results" : "./results"
    CSV.write("$output_dir/results_$(dataset_name)_$(mode).csv", df)
    
    return df
end

# Load data
if mode == "LS"
    balanced, binary, alphas, _, _, _, ε, kn, noise, percentage_of_known_labels, num_trials, dataset_name, data_type = prepare_config_data(data[mode])
    features, y, A, DG_isqrt = load_data(dataset_name, kn, noise, nothing, mode, binary)
    
    # Run both original and perturbed analysis
    run_analysis(mode, data, y, features, A, DG_isqrt, use_perturbed=false)
    run_analysis(mode, data, y, features, A, DG_isqrt, use_perturbed=true)
    
elseif mode == "HOLS"
    balanced, binary, alphas, betas, weight_function, mixing_functions, ε, kn, noise, percentage_of_known_labels, num_trials, dataset_name, data_type = prepare_config_data(data[mode])
    features, y, A, DG_isqrt, T, DH_isqrt, B = load_data(dataset_name, kn, noise, weight_function, mode, binary)
    
    # Run both original and perturbed analysis
    run_analysis(mode, data, y, features, A, DG_isqrt, T, DH_isqrt, B, use_perturbed=false)
    run_analysis(mode, data, y, features, A, DG_isqrt, T, DH_isqrt, B, use_perturbed=true)
    
elseif mode == "both"
    balanced, binary, alphas, betas, weight_function, mixing_functions, ε, kn, noise, percentage_of_known_labels, num_trials, dataset_name, data_type = prepare_config_data(data[mode])
    features, y, A, DG_isqrt, T, DH_isqrt, B = load_data(dataset_name, kn, noise, weight_function, mode, binary)
    
    # Run both original and perturbed analysis
    run_analysis(mode, data, y, features, A, DG_isqrt, T, DH_isqrt, B, use_perturbed=false)
    run_analysis(mode, data, y, features, A, DG_isqrt, T, DH_isqrt, B, use_perturbed=true)
end