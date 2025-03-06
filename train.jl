include("src/utils.jl")
include("src/tensors.jl")
include("src/CV_helpers.jl")
include("src/functions.jl")
include("src/labelspreading.jl")


using YAML


data = YAML.load(open("config.yml"))


mode = data["mode"]


if !isdir("./results")
   mkdir("./results")
end


colnames = [:method_name, :size, :knn, :percentage_of_known_labels, :α, :β, :balanced, :noise, :acc, :prec, :rec]


if mode == "LS"
   balanced, binary, alphas, _, _, _, ε, kn, noise, percentage_of_known_labels, num_trials, dataset_name, data_type = prepare_config_data(data[mode])
   weight_function = nothing
   features, y, A, DG_isqrt = load_data(dataset_name, kn, noise, weight_function, mode, binary)
   data = analyze_dataset_LS(dataset_name,num_trials,
                                       alphas,
                                       kn,
                                       A, DG_isqrt,
                                       percentage_of_known_labels,
                                       y,
                                       balanced,
                                       ε)
   df = DataFrame(data)
   println("DataFrame columns before rename: ", names(df))
   println("Attempting to rename to: ", colnames)
   rename!(df, colnames)
   CSV.write("./results/results_$(dataset_name)_$(mode).csv", df)


elseif (mode == "HOLS")
    balanced, binary, alphas, betas, weight_function, mixing_functions, ε, kn, noise, percentage_of_known_labels, num_trials, dataset_name, data_type = prepare_config_data(data[mode])
    features, y, A, DG_isqrt, T, DH_isqrt, B = load_data(dataset_name, kn, noise, weight_function, mode, binary)
    results = analyze_dataset_HOLS(dataset_name,num_trials,
                                        kn,
                                        A, DG_isqrt, T, DH_isqrt,B,φ,
                                        mixing_functions,
                                        percentage_of_known_labels,
                                        y,
                                        balanced,
                                        alphas,
                                        betas,
                                        ε)
 
 
    # Create DataFrame from the results
    df = DataFrame(
        method_name = [r[2] for r in results],  # dataset_name used as method_name
        size = [r[3] for r in results],
        knn = [r[4] for r in results],
        percentage_of_known_labels = [r[8] for r in results],
        α = [r[6] for r in results],
        β = [r[7] for r in results],
        balanced = [r[9] for r in results],
        noise = repeat([noise], length(results)),
        acc = [r[10] for r in results],
        prec = [r[11] for r in results],
        rec = [r[12] for r in results]
    )
   
    CSV.write("./results/results_$(dataset_name)_$(mode).csv", df)
 
 
 elseif (mode == "both")
    balanced, binary, alphas, betas, weight_function, mixing_functions, ε, kn, noise, percentage_of_known_labels, num_trials, dataset_name, data_type = prepare_config_data(data[mode])
   features, y, A, DG_isqrt, T, DH_isqrt, B = load_data(dataset_name, kn, noise, weight_function, mode, binary)
   data = analyze_dataset(dataset_name,num_trials,
                                           kn,
                                           A, DG_isqrt, T, DH_isqrt, B,φ,
                                           mixing_functions,
                                           percentage_of_known_labels,
                                           y,
                                           balanced,
                                           alphas,
                                           betas,
                                           ε)


   df = DataFrame(data)
   println("DataFrame columns before rename: ", names(df))
   println("Attempting to rename to: ", colnames)
   rename!(df, colnames)
   CSV.write("./results/results_$(dataset_name)_$(mode).csv", df)
end