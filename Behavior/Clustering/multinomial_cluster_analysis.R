#!/usr/bin/env Rscript
# Cluster Analysis using Multinomial GEE
# Formula: cluster ~ group + inclusion_exclusion + group * inclusion_exclusion

library(multgee)
library(tidyverse)

# Create output directory
output_dir <- "results/Behavior/Clustering/multgee_analysis"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

print("=== MULTINOMIAL GEE CLUSTER ANALYSIS ===")

# Load data
df_clusters <- read.csv("results/Behavior/Clustering/clustering_hierarchical.csv")
df_metadata <- read.csv("metadata_experiment.csv")

# Merge datasets
df_metadata$subject_id <- df_metadata$subj
df_merged <- merge(df_clusters, df_metadata, by = "subject_id", all.x = TRUE)

# Map cluster and group labels
df_merged$Cluster <- factor(df_merged$Cluster, 
                           levels = c(0, 1, 2),
                           labels = c("Low", "Self-Centered", "Positive-Future-Oriented"))

# Explicitly set "Low" as reference category
df_merged$Cluster <- relevel(df_merged$Cluster, ref = "Low")

df_merged$group <- factor(df_merged$group,
                         levels = c(1, 2), 
                         labels = c("Controls", "Risk of Depression"))

print(paste("Total observations:", nrow(df_merged)))
print(paste("Unique subjects:", length(unique(df_merged$subject_id))))
print(paste("Cluster levels (reference first):", paste(levels(df_merged$Cluster), collapse = ", ")))

# Create inclusion/exclusion mapping
create_ie_mapping <- function(metadata) {
  order_col <- names(metadata)[grepl("order", names(metadata), ignore.case = TRUE)][1]
  subject_to_order <- setNames(metadata[[order_col]], metadata$subj)
  
  mapping <- list()
  for (subj in names(subject_to_order)) {
    order <- subject_to_order[subj]
    if (!is.na(order)) {
      if (order == "IE") {
        mapping[[paste(subj, "Sart2", sep = "_")]] <- "inclusion"
        mapping[[paste(subj, "Sart4", sep = "_")]] <- "exclusion"
      } else if (order == "EI") {
        mapping[[paste(subj, "Sart2", sep = "_")]] <- "exclusion"
        mapping[[paste(subj, "Sart4", sep = "_")]] <- "inclusion"
      }
    }
  }
  return(mapping)
}

# Apply inclusion/exclusion mapping
ie_mapping <- create_ie_mapping(df_metadata)

# Filter to Sart2/Sart4 and add inclusion/exclusion
df_filtered <- df_merged %>%
  filter(task %in% c("Sart2", "Sart4")) %>%
  mutate(
    key = paste(subject_id, task, sep = "_"),
    inclusion_exclusion = sapply(key, function(k) ie_mapping[[k]]),
    inclusion_exclusion = factor(inclusion_exclusion, levels = c("inclusion", "exclusion"))
  ) %>%
  filter(!is.na(inclusion_exclusion))

print(paste("Final dataset:", nrow(df_filtered), "observations"))
print(paste("Subjects:", length(unique(df_filtered$subject_id))))

# Data summary
print("=== DATA SUMMARY ===")
print(table(df_filtered$Cluster))
print(table(df_filtered$group))
print(table(df_filtered$inclusion_exclusion))

# Prepare for GEE
df_gee <- df_filtered %>%
  mutate(
    # Keep cluster as factor for proper reference category handling
    cluster_factor = Cluster,
    group_binary = as.numeric(group) - 1,
    ie_binary = as.numeric(inclusion_exclusion) - 1,
    participant_id = factor(subject_id)
  ) %>%
  arrange(participant_id)

# Print cluster mapping for reference
print("=== CLUSTER MAPPING ===")
print("Reference category: Low")
print("Comparisons will be:")
print("- Self-Centered vs Low")
print("- Positive-Future-Oriented vs Low")

# Save data summary
summary_stats <- data.frame(
  variable = c("observations", "subjects", "controls", "risk_depression", 
              "inclusion", "exclusion", "low", "self_centered", "positive_future",
              "controls_inclusion", "controls_exclusion", "risk_inclusion", "risk_exclusion",
              "controls_low", "controls_self_centered", "controls_positive_future",
              "risk_low", "risk_self_centered", "risk_positive_future",
              "inclusion_low", "inclusion_self_centered", "inclusion_positive_future",
              "exclusion_low", "exclusion_self_centered", "exclusion_positive_future"),
  count = c(nrow(df_filtered), length(unique(df_filtered$subject_id)),
          sum(df_filtered$group == "Controls"), sum(df_filtered$group == "Risk of Depression"),
          sum(df_filtered$inclusion_exclusion == "inclusion"), sum(df_filtered$inclusion_exclusion == "exclusion"),
          sum(df_filtered$Cluster == "Low"), sum(df_filtered$Cluster == "Self-Centered"), 
          sum(df_filtered$Cluster == "Positive-Future-Oriented"),
          # Group × Inclusion/Exclusion interactions
          sum(df_filtered$group == "Controls" & df_filtered$inclusion_exclusion == "inclusion"),
          sum(df_filtered$group == "Controls" & df_filtered$inclusion_exclusion == "exclusion"),
          sum(df_filtered$group == "Risk of Depression" & df_filtered$inclusion_exclusion == "inclusion"),
          sum(df_filtered$group == "Risk of Depression" & df_filtered$inclusion_exclusion == "exclusion"),
          # Group × Cluster interactions
          sum(df_filtered$group == "Controls" & df_filtered$Cluster == "Low"),
          sum(df_filtered$group == "Controls" & df_filtered$Cluster == "Self-Centered"),
          sum(df_filtered$group == "Controls" & df_filtered$Cluster == "Positive-Future-Oriented"),
          sum(df_filtered$group == "Risk of Depression" & df_filtered$Cluster == "Low"),
          sum(df_filtered$group == "Risk of Depression" & df_filtered$Cluster == "Self-Centered"),
          sum(df_filtered$group == "Risk of Depression" & df_filtered$Cluster == "Positive-Future-Oriented"),
          # Inclusion/Exclusion × Cluster interactions
          sum(df_filtered$inclusion_exclusion == "inclusion" & df_filtered$Cluster == "Low"),
          sum(df_filtered$inclusion_exclusion == "inclusion" & df_filtered$Cluster == "Self-Centered"),
          sum(df_filtered$inclusion_exclusion == "inclusion" & df_filtered$Cluster == "Positive-Future-Oriented"),
          sum(df_filtered$inclusion_exclusion == "exclusion" & df_filtered$Cluster == "Low"),
          sum(df_filtered$inclusion_exclusion == "exclusion" & df_filtered$Cluster == "Self-Centered"),
          sum(df_filtered$inclusion_exclusion == "exclusion" & df_filtered$Cluster == "Positive-Future-Oriented"))
)
write.csv(summary_stats, file.path(output_dir, "data_summary.csv"), row.names = FALSE)

# Function to run GEE analysis
run_gee <- function(formula_str, model_name) {
  print(paste("=== MODEL:", model_name, "==="))
  print(paste("Formula:", formula_str))

  # print(df_gee)
  
  tryCatch({
    model <- nomLORgee(as.formula(formula_str), 
                      id = participant_id, 
                      data = df_gee, 
                      LORstr = "independence")
    
    summary_model <- summary(model)
    # print(summary_model)
    coefs <- summary_model$coefficients
    print(coefs)
    
    # Create results table
    results <- data.frame(
      coefficient = rownames(coefs),
      estimate = coefs[, "Estimate"],
      se = coefs[, "san.se"],
      z_value = coefs[, "san.z"],
      p_value = coefs[, "Pr(>|san.z|)"],
      stringsAsFactors = FALSE
    )
    
    # Classify effect types
    results$effect_type <- "intercept"
    results$effect_type[grepl("group_binary:", results$coefficient) & 
                      !grepl("ie_binary", results$coefficient)] <- "main_group"
    results$effect_type[grepl("ie_binary:", results$coefficient) & 
                      !grepl("group_binary", results$coefficient)] <- "main_ie"
    results$effect_type[grepl("group_binary:ie_binary:", results$coefficient)] <- "interaction"
    
    # Add cluster comparisons - based on coefficient patterns
    results$cluster_comparison <- NA
    results$cluster_comparison[grepl(":1$", results$coefficient)] <- "Self-Centered_vs_Low"
    results$cluster_comparison[grepl(":2$", results$coefficient)] <- "Positive-Future-Oriented_vs_Low"
    
    # Add more interpretable coefficient names
    results$coefficient_clean <- results$coefficient
    
    # Clean up coefficient names with specific mappings (order matters!)
    # Handle interaction effects first (most specific)
    results$coefficient_clean <- gsub("group_binary:ie_binary:1", "GroupXInclusionExclusion_interaction_Self-Centered_vs_Low", results$coefficient_clean)
    results$coefficient_clean <- gsub("group_binary:ie_binary:2", "GroupXInclusionExclusion_interaction_Positive-Future-Oriented_vs_Low", results$coefficient_clean)
    
    # Handle main effects
    results$coefficient_clean <- gsub("group_binary:1", "Group_effect_Self-Centered_vs_Low", results$coefficient_clean)
    results$coefficient_clean <- gsub("group_binary:2", "Group_effect_Positive-Future-Oriented_vs_Low", results$coefficient_clean)
    results$coefficient_clean <- gsub("ie_binary:1", "InclusionExclusion_effect_Self-Centered_vs_Low", results$coefficient_clean)
    results$coefficient_clean <- gsub("ie_binary:2", "InclusionExclusion_effect_Positive-Future-Oriented_vs_Low", results$coefficient_clean)
    
    # Handle intercepts
    results$coefficient_clean <- gsub("beta10", "Intercept_Self-Centered_vs_Low", results$coefficient_clean)
    results$coefficient_clean <- gsub("beta20", "Intercept_Positive-Future-Oriented_vs_Low", results$coefficient_clean)
    
    # Significance flags
    results$significant <- results$p_value < 0.05
    
    # Bonferroni correction for non-intercepts
    non_intercept <- results[results$effect_type != "intercept", ]
    if (nrow(non_intercept) > 0) {
      non_intercept$p_bonferroni <- p.adjust(non_intercept$p_value, method = "bonferroni")
      non_intercept$sig_corrected <- non_intercept$p_bonferroni < 0.05
      
      print(paste("Significant effects (uncorrected):", sum(non_intercept$significant)))
      print(paste("Significant effects (Bonferroni):", sum(non_intercept$sig_corrected)))
      
      # Save results
      write.csv(results, file.path(output_dir, paste0(model_name, "_results.csv")), row.names = FALSE)
      write.csv(non_intercept, file.path(output_dir, paste0(model_name, "_effects.csv")), row.names = FALSE)
      
      # Display significant effects
      sig_effects <- non_intercept[non_intercept$sig_corrected, ]
      if (nrow(sig_effects) > 0) {
        print("SIGNIFICANT EFFECTS (Bonferroni corrected):")
        for (i in 1:nrow(sig_effects)) {
          cluster_info <- if(!is.na(sig_effects$cluster_comparison[i])) {
            paste0(sig_effects$cluster_comparison[i], " - ")
          } else {
            ""
          }
          cat(paste0("- ", cluster_info, sig_effects$coefficient_clean[i], 
                    " (", sig_effects$effect_type[i], "): ",
                    "β=", round(sig_effects$estimate[i], 3), 
                    ", p=", round(sig_effects$p_bonferroni[i], 4), "\n"))
        }
      } else {
        print("No significant effects after Bonferroni correction.")
      }
    }
    
    return(list(model = model, results = results))
    
  }, error = function(e) {
    print(paste("Error in", model_name, ":", e$message))
    return(NULL)
  })
}

# Run all models
print(paste(rep("=", 60), collapse = ""))
model1 <- run_gee("cluster_factor ~ group_binary", "model1_group")

print(paste(rep("=", 60), collapse = ""))
model2 <- run_gee("cluster_factor ~ ie_binary", "model2_ie")

print(paste(rep("=", 60), collapse = ""))
model3 <- run_gee("cluster_factor ~ group_binary + ie_binary", "model3_additive")

print(paste(rep("=", 60), collapse = ""))
model4 <- run_gee("cluster_factor ~ group_binary * ie_binary", "model4_interaction")

print(paste(rep("=", 60), collapse = ""))
print("ANALYSIS COMPLETE")
print(paste("Results saved to:", output_dir))

print("")
print("=== INTERPRETATION GUIDE ===")
print("Reference category: Low cluster")
print("All effects are compared against the Low cluster:")
print("- ':1' coefficients = Self-Centered vs Low")
print("- ':2' coefficients = Positive-Future-Oriented vs Low")
print("")
print("Effect types:")
print("- beta10/beta20 = Intercepts for each comparison")
print("- group_binary = Group effect (Risk of Depression vs Controls)")
print("- ie_binary = Inclusion/Exclusion effect")
print("- group_binary:ie_binary = Group x Inclusion/Exclusion interaction")
print("")
print("Positive coefficients indicate higher probability of the non-reference category") 