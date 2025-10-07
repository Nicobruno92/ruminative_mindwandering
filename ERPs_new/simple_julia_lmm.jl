#!/usr/bin/env julia

"""
Advanced Julia temporal analysis using Unfold.jl

This script provides temporal Linear Mixed Model analysis using Unfold.jl
for optimized mass univariate EEG analysis with proper temporal modeling.

Usage:
    julia --startup-file=no simple_julia_lmm.jl --input data.csv --output results/

Author: Nicolas Bruno
"""

println("📦 Loading Julia packages...")
flush(stdout)

using DataFrames
println("  ✅ DataFrames loaded")
flush(stdout)

using CSV
println("  ✅ CSV loaded")
flush(stdout)

using Unfold
println("  ✅ Unfold loaded")
flush(stdout)

using MixedModels
println("  ✅ MixedModels loaded")
flush(stdout)

using StatsModels
println("  ✅ StatsModels loaded")
flush(stdout)

using CategoricalArrays
println("  ✅ CategoricalArrays loaded")
flush(stdout)

using Statistics
println("  ✅ Statistics loaded")
flush(stdout)

using Distributions
println("  ✅ Distributions loaded")
flush(stdout)

using ArgParse
println("  ✅ ArgParse loaded")
flush(stdout)

using Printf
println("  ✅ Printf loaded")
flush(stdout)

println("✅ All packages loaded successfully")
flush(stdout)

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--input", "-i"
            help = "Input CSV data file"
            required = true
        "--output", "-o"
            help = "Output directory"
            required = true
        "--formula", "-f"
            help = "LMM formula"
            default = "amplitude ~ condition_code + baseline_centered"
        "--alpha"
            help = "Significance level"
            arg_type = Float64
            default = 0.05
    end
    return parse_args(s)
end

function load_data(file_path::String)
    println("📂 Loading data: $file_path")
    flush(stdout)
    
    # Check if file exists
    if !isfile(file_path)
        error("Input file does not exist: $file_path")
    end
    
    println("📂 File exists, reading CSV...")
    flush(stdout)
    
    df = CSV.read(file_path, DataFrame)
    println("✅ Loaded $(nrow(df)) rows, $(ncol(df)) columns")
    println("📊 Columns: $(names(df))")
    flush(stdout)
    
    # Check for required columns for continuous temporal data
    required_cols = ["subject", "condition", "roi", "trial", "time_point", "amplitude"]
    missing_cols = setdiff(required_cols, names(df))
    if !isempty(missing_cols)
        error("Missing required columns for Unfold analysis: $missing_cols")
    end
    
    # Print data preview for continuous temporal data
    println("📊 Continuous temporal data preview:")
    println("  - Subjects: $(unique(df.subject)[1:min(5, end)])")
    println("  - Conditions: $(unique(df.condition))")
    println("  - ROIs: $(unique(df.roi))")
    println("  - Trials: $(length(unique(df.trial))) total trials")
    println("  - Time point range: $(minimum(df.time_point)) to $(maximum(df.time_point)) seconds")
    println("  - Time points per trial: $(length(unique(df.time_point)))")
    println("  - Amplitude range: $(minimum(df.amplitude)) to $(maximum(df.amplitude))")
    
    # Convert categorical variables
    df.subject = categorical(string.(df.subject))
    df.condition = categorical(string.(df.condition))
    df.roi = categorical(string.(df.roi))
    df.trial = categorical(string.(df.trial))
    
    # Create condition coding (onTask=0, offTask=1)  
    df.condition_code = Float64.(df.condition .== "offTask")
    println("📊 Condition coding: $(sum(df.condition_code .== 1)) offTask, $(sum(df.condition_code .== 0)) onTask")
    
    # Compute baseline for each trial (using pre-stimulus period)
    df.baseline_centered = zeros(Float64, nrow(df))
    
    # Group by trial to compute baseline per trial
    for trial_group in groupby(df, [:subject, :roi, :trial])
        if nrow(trial_group) == 0
            continue
        end
        
        # Find baseline period (typically negative times, e.g., -0.2 to 0.0)
        baseline_mask = trial_group.time_point .< 0.0
        if any(baseline_mask)
            baseline_data = trial_group.amplitude[baseline_mask]
            if length(baseline_data) > 0
                baseline_mean = mean(baseline_data)
                # Apply baseline correction to this trial
                trial_indices = findall(
                    (df.subject .== trial_group.subject[1]) .& 
                    (df.roi .== trial_group.roi[1]) .& 
                    (df.trial .== trial_group.trial[1])
                )
                df.baseline_centered[trial_indices] = df.amplitude[trial_indices] .- baseline_mean
            end
        else
            # No baseline period found, use raw amplitude
            trial_indices = findall(
                (df.subject .== trial_group.subject[1]) .& 
                (df.roi .== trial_group.roi[1]) .& 
                (df.trial .== trial_group.trial[1])
            )
            df.baseline_centered[trial_indices] = df.amplitude[trial_indices]
        end
    end
    
    println("✅ Continuous temporal data prepared for Unfold analysis")
    println("📊 $(nrow(df)) data points across $(length(unique(df.trial))) trials")
    return df
end

function run_unfold_analysis(df::DataFrame, formula_str::String, alpha::Float64)
    println("🧠 Running PROPER Unfold.jl temporal analysis with continuous data...")
    println("📋 Formula: $formula_str")
    println("📊 Input data: $(nrow(df)) data points")
    println("📊 Unique ROIs: $(length(unique(df.roi)))")
    println("📊 Unique trials: $(length(unique(df.trial)))")
    println("📊 Unique time points: $(length(unique(df.time_point)))")
    println("📊 Unique subjects: $(length(unique(df.subject)))")
    println("📊 Unique conditions: $(unique(df.condition))")
    flush(stdout)
    
    results = DataFrame(
        roi = String[],
        time_point = Float64[],
        coefficient = Float64[],
        std_error = Float64[],
        t_value = Float64[],
        p_value = Float64[],
        ci_lower = Float64[],
        ci_upper = Float64[],
        n_subjects = Int[],
        n_observations = Int[]
    )
    
    # Convert formula string to Unfold format
    # Unfold uses: 0 ~ 1 + predictors + (1|subject) format
    unfold_formula_str = if occursin("(1|subject", formula_str)
        replace(formula_str, "amplitude ~" => "0 ~ 1 +")
    else
        replace(formula_str, "amplitude ~" => "0 ~ 1 +") * " + (1|subject)"
    end
    println("📋 Unfold formula: $unfold_formula_str")
    
    # Parse Unfold formula
    formula = @eval(@formula($(Meta.parse(unfold_formula_str))))
    
    # Process each ROI separately (Unfold works on single channel/ROI data)
    rois = unique(df.roi)
    println("📊 Processing $(length(rois)) ROIs with Unfold...")
    
    roi_count = 0
    
    for roi in rois
        roi_count += 1
        println("🔧 Processing ROI $roi_count/$(length(rois)): $roi")
        flush(stdout)
        
        # Filter data for this ROI
        roi_data = filter(row -> row.roi == roi, df)
        
        # Check if we have enough data for this ROI
        n_subjects = length(unique(roi_data.subject))
        n_conditions = length(unique(roi_data.condition))
        n_trials = length(unique(roi_data.trial))
        n_timepoints = length(unique(roi_data.time_point))
        
        println("  📊 ROI $roi: $n_subjects subjects, $n_conditions conditions, $n_trials trials, $n_timepoints timepoints")
        
        if n_subjects < 3 || n_conditions < 2 || n_trials < 10 || n_timepoints < 10
            println("  ⚠️  Skipping $roi: insufficient data")
            continue
        end
        
        try
            # Prepare data for Unfold
            # Unfold expects: events (DataFrame), data (matrix), times (vector)
            
            # ========================================================================
            # PREPARE DATA FOR UNFOLD.JL CORRECTLY
            # ========================================================================
            
            # Get sorted unique time points and trials
            times = sort(unique(roi_data.time_point))
            trials = sort(unique(roi_data.trial))
            
            println("  📊 Preparing Unfold data structure...")
            println("    - Time range: $(times[1]) to $(times[end]) s")
            println("    - $(length(times)) time points")
            println("    - $(length(trials)) trials")
            
            # Create EVENTS dataframe (one row per trial)
            events_data = []
            
            # Create DATA matrix (trials × timepoints)
            data_matrix = zeros(Float64, length(trials), length(times))
            
            # Process each trial
            
            for (trial_idx, trial) in enumerate(trials)
                # Get data for this trial
                trial_data = filter(row -> row.trial == trial, roi_data)
                
                if nrow(trial_data) == 0
                    continue
                end
                
                # Create event row (trial-level information)
                # Get trial metadata (should be same for all timepoints in this trial)
                trial_subject = string(trial_data.subject[1])
                trial_condition = string(trial_data.condition[1])
                trial_condition_code = trial_data.condition_code[1]
                
                event_row = (
                    trial = trial_idx,
                    subject = trial_subject,
                    condition = trial_condition,
                    condition_code = trial_condition_code
                )
                push!(events_data, event_row)
                
                # Fill data matrix for this trial
                for (time_idx, time_point) in enumerate(times)
                    # Find amplitude at this time point for this trial
                    time_rows = filter(row -> row.time_point == time_point, trial_data)
                    if nrow(time_rows) > 0
                        # Use baseline-corrected amplitude
                        data_matrix[trial_idx, time_idx] = time_rows.baseline_centered[1]
                    else
                        # Missing data - could interpolate or use NaN
                        data_matrix[trial_idx, time_idx] = NaN
                    end
                end
            end
            
            if length(events_data) == 0
                println("  ⚠️  No events created for $roi")
                continue
            end
            
            # Convert to required formats
            events_df = DataFrame(events_data)
            events_df.subject = categorical(events_df.subject)
            
            println("  � Unfold input: $(nrow(events_df)) trials, $(length(times)) timepoints")
            
            # Fit Unfold model
            println("  🔧 Fitting Unfold model...")
            unfold_model = fit(UnfoldModel, formula, events_df, data_matrix, times)
            println("  ✅ Unfold model fitted successfully")
            
            # Extract results - Unfold returns coefficients per time point
            coef_results = coeftable(unfold_model)
            betas = Unfold.coef(unfold_model)  # Get coefficient matrix (predictors x timepoints)
            stderrs = Unfold.stderror(unfold_model)  # Get standard errors
            
            # Get predictor names
            predictor_names = coefnames(unfold_model)
            condition_idx = findfirst(x -> occursin("condition_code", string(x)), predictor_names)
            
            if condition_idx !== nothing
                # Extract coefficients for condition_code across all time points
                condition_betas = betas[condition_idx, :]  # Row for condition_code
                condition_ses = stderrs[condition_idx, :]  # Standard errors
                
                # Process results for each time point
                for (time_idx, time_point) in enumerate(times)
                    coef = condition_betas[time_idx]
                    se = condition_ses[time_idx]
                    t_val = coef / se  # t-statistic
                    p_val = 2 * (1 - cdf(TDist(max(1, nrow(events_df) - length(predictor_names))), abs(t_val)))  # p-value
                    
                    # Confidence intervals
                    t_crit = 1.96  # approximate for large samples
                    ci_lower = coef - t_crit * se
                    ci_upper = coef + t_crit * se
                    
                    push!(results, (
                        string(roi),
                        Float64(time_point),
                        coef,
                        se,
                        t_val,
                        p_val,
                        ci_lower,
                        ci_upper,
                        n_subjects,
                        nrow(events_df)  # number of trials
                    ))
                end
                
                println("  ✅ Extracted results for $(length(times)) time points")
            else
                println("  ⚠️  No condition coefficient found in predictors: $predictor_names")
            end
            
        catch e
            println("  ❌ Unfold analysis failed for $roi: $e")
            println("  � Error details: $(typeof(e))")
            
            # No fallback - if Unfold fails, we want to know why
            continue
        end
    end
    
    # Add significance column
    results.significant = results.p_value .< alpha
    
    println("📊 Unfold analysis complete:")
    println("  - Processed: $(length(rois)) ROIs")
    println("  - Results: $(nrow(results)) time points")
    println("  - Significant: $(sum(results.significant)) effects")
    
    return results
end

function save_results(results::DataFrame, output_dir::String)
    mkpath(output_dir)
    
    # Save main results
    results_file = joinpath(output_dir, "simple_lmm_results.csv")
    CSV.write(results_file, results)
    println("✅ Results saved: $results_file")
    
    # Print summary
    println("\n📊 SUMMARY:")
    println("Total time points analyzed: $(nrow(results))")
    println("ROIs: $(length(unique(results.roi)))")
    println("Significant effects: $(sum(results.significant))")
    
    if sum(results.significant) > 0
        println("\nSignificant effects by ROI:")
        for roi in unique(results.roi)
            roi_results = filter(row -> row.roi == roi, results)
            n_sig = sum(roi_results.significant)
            if n_sig > 0
                println("  $roi: $n_sig/$(nrow(roi_results)) time points")
            end
        end
    end
end

function main()
    println("🔬 Unfold.jl Temporal LMM Analysis")
    println("=" ^ 50)
    
    # Flush stdout to ensure it's printed immediately
    flush(stdout)
    
    println("📋 Parsing command line arguments...")
    flush(stdout)
    
    args = parse_commandline()
    
    println("📋 Arguments parsed successfully:")
    println("  - input: $(args["input"])")
    println("  - output: $(args["output"])")
    println("  - formula: $(args["formula"])")
    println("  - alpha: $(args["alpha"])")
    flush(stdout)
    
    try
        # Load and prepare data
        println("📋 About to load data...")
        flush(stdout)
        df = load_data(args["input"])
        
        # Run analysis
        println("📋 About to run Unfold analysis...")
        flush(stdout)
        results = run_unfold_analysis(df, args["formula"], args["alpha"])
        
        # Save results
        println("📋 About to save results...")
        flush(stdout)
        save_results(results, args["output"])
        
        println("\n🎉 Analysis completed successfully!")
        flush(stdout)
        
    catch e
        println("❌ Analysis failed: $e")
        flush(stdout)
        exit(1)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

