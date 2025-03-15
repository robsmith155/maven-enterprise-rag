import os
import re
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import hsv_to_rgb


def plot_consecutive_recall_results(results, output_path=None):
    """
    Plot evaluation results focusing on consecutive coverage recall and MRR at different k values.
    
    Args:
        results: Evaluation results from evaluate_pipeline_with_consecutive_recall
        output_path: Path to save the plot (optional)
    """
    
    # Prepare data for plotting
    k_values = sorted(results["consecutive_recall_at_k"].keys())
    recall_values = [results["consecutive_recall_at_k"][k] for k in k_values]
    mrr_values = [results["mrr_at_k"][k] for k in k_values]
    
    # Create figure with two subplots side by side
    plt.figure(figsize=(15, 6))
    
    # Plot Consecutive Recall@k
    plt.subplot(1, 2, 1)
    plt.plot(k_values, recall_values, marker='o', color='blue', linewidth=2)
    plt.xlabel('k')
    plt.ylabel('Consecutive Coverage Recall@k')
    plt.title('Average Consecutive Coverage Recall@k')
    plt.grid(True)
    
    # Plot MRR@k
    plt.subplot(1, 2, 2)
    plt.plot(k_values, mrr_values, marker='s', color='orange', linewidth=2)
    plt.xlabel('k')
    plt.ylabel('MRR@k')
    plt.title('Average MRR@k')
    plt.grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved plot to {output_path}")
    
    plt.close()


def plot_base_configuration_comparison(base_config_results, k_values, output_dir, base_key):
    """
    Generate comparison plots for all variants within a single base configuration.
    
    Args:
        base_config_results: Dictionary with results for configurations sharing the same base
        k_values: List of k values used in evaluation
        output_dir: Directory to save the plots
        base_key: The base configuration key for title display
    """    
    # Sort k values
    k_values = sorted([int(k) for k in k_values])
    
    # Define the four possible configurations
    config_types = {
        "standard": {"reranking": False, "hybrid": False, "label": "Standard", "color": "blue", "marker": "o"},
        "reranking": {"reranking": True, "hybrid": False, "label": "Reranking Only", "color": "green", "marker": "o"},
        "hybrid": {"reranking": False, "hybrid": True, "label": "Hybrid Only", "color": "red", "marker": "s"},
        "reranking_hybrid": {"reranking": True, "hybrid": True, "label": "Reranking + Hybrid", "color": "purple", "marker": "s"}
    }
    
    # Map each result to one of the four configuration types
    mapped_results = {}
    for config_name, results in base_config_results.items():
        use_reranking = 'with_reranking' in config_name
        use_hybrid = 'hybrid' in config_name
        
        # Determine which of the four types this is
        if use_reranking and use_hybrid:
            config_type = "reranking_hybrid"
        elif use_reranking:
            config_type = "reranking"
        elif use_hybrid:
            config_type = "hybrid"
        else:
            config_type = "standard"
        
        mapped_results[config_type] = results
    
    # Create figure for recall comparison
    plt.figure(figsize=(12, 7))
    
    # Plot each configuration type that exists in our results
    for config_type, config_info in config_types.items():
        if config_type in mapped_results:
            results = mapped_results[config_type]
            
            # Get data
            recall_values = [results["consecutive_recall_at_k"][str(k)] for k in k_values]
            
            # Plot with appropriate style
            plt.plot(k_values, recall_values, 
                     marker=config_info["marker"], 
                     color=config_info["color"], 
                     linewidth=2, 
                     label=config_info["label"])
    
    plt.xlabel('k', fontsize=12)
    plt.ylabel('Consecutive Coverage Recall@k', fontsize=12)
    plt.title(f'Comparison of Retrieval Strategies - {base_key}', fontsize=14)
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, "retrieval_strategies_comparison.png"))
    plt.close()
    
    # Create figure for MRR comparison
    plt.figure(figsize=(12, 7))
    
    # Plot each configuration type that exists in our results
    for config_type, config_info in config_types.items():
        if config_type in mapped_results:
            results = mapped_results[config_type]
            
            # Get data
            mrr_values = [results["mrr_at_k"][str(k)] for k in k_values]
            
            # Plot with appropriate style
            plt.plot(k_values, mrr_values, 
                     marker=config_info["marker"], 
                     color=config_info["color"], 
                     linewidth=2, 
                     label=config_info["label"])
    
    plt.xlabel('k', fontsize=12)
    plt.ylabel('MRR@k', fontsize=12)
    plt.title(f'Comparison of MRR@k - {base_key}', fontsize=14)
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, "mrr_strategies_comparison.png"))
    plt.close()
    
    # Create a bar chart comparing performance at specific k values
    selected_k_values = [5, 20] if 20 in k_values else [5, 10]
    
    for k in selected_k_values:
        plt.figure(figsize=(12, 7))
        
        # Prepare data
        config_labels = []
        recall_values = []
        mrr_values = []
        bar_colors = []
        
        for config_type, config_info in config_types.items():
            if config_type in mapped_results:
                results = mapped_results[config_type]
                config_labels.append(config_info["label"])
                recall_values.append(results["consecutive_recall_at_k"][str(k)])
                mrr_values.append(results["mrr_at_k"][str(k)])
                bar_colors.append(config_info["color"])
        
        # Set up bar positions
        x = range(len(config_labels))
        width = 0.35
        
        # Create bars with consistent colors
        plt.bar([i - width/2 for i in x], recall_values, width, label='Recall', color='blue', alpha=0.7)
        plt.bar([i + width/2 for i in x], mrr_values, width, label='MRR', color='orange', alpha=0.7)
        
        plt.xlabel('Retrieval Strategy', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title(f'Recall and MRR at k={k} - {base_key}', fontsize=14)
        plt.xticks(x, config_labels)
        plt.legend()
        plt.tight_layout()
        plt.grid(True, axis='y')
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, f"strategy_comparison_k{k}.png"))
        plt.close()


def generate_comparison_plots(all_results, k_values, output_dir):
    """
    Generate comparison plots for all configurations with improved visualization.
    Colors are assigned based on base configurations (chunking method + embedding model),
    while variations use different line styles.
    
    Args:
        all_results: Dictionary with results for all configurations
        k_values: List of k values for Recall@k and MRR@k
        output_dir: Directory to save plots
    """
    
    # Set up plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(16, 10))
    
    # Extract base configurations (unique combinations of chunking method and embedding model)
    base_configs = set()
    for config_name in all_results.keys():
        # Extract chunking method
        chunking_match = re.search(r'(token|semantic|late)', config_name)
        chunking = chunking_match.group(1) if chunking_match else "unknown"
        
        # Extract embedding model
        embedding_match = re.search(r'(bge-small-en-v1\.5|bge-large-en-v1\.5|bge-base-en-v1\.5|e5-small-v2)', config_name)
        embedding = embedding_match.group(1) if embedding_match else "unknown"
        
        # Extract chunk size
        size_match = re.search(r'chunk(\d+)', config_name)
        size = size_match.group(1) if size_match else "0"
        
        # Create base config key
        base_config = f"{chunking}_{size}_{embedding}"
        base_configs.add(base_config)
    
    # Create color map for base configurations
    base_configs = sorted(list(base_configs))
    num_base_configs = len(base_configs)
    
    # Generate distinct colors using a colormap
    cmap = cm.get_cmap('tab20' if num_base_configs <= 20 else 'viridis')
    base_config_colors = {}
    
    for i, base_config in enumerate(base_configs):
        color_idx = i / max(1, num_base_configs - 1)  # Normalize to [0, 1]
        base_config_colors[base_config] = cmap(color_idx)
    
    # Define line styles for variations
    variation_styles = {
        # Format: (reranking, hybrid, tables) -> linestyle
        (False, False, False): '-',      # Solid line: standard
        (True, False, False): '--',      # Dashed: reranking only
        (False, True, False): '-.',      # Dash-dot: hybrid only
        (True, True, False): ':',        # Dotted: reranking + hybrid
        (False, False, True): (0, (3, 1, 1, 1)),  # Dash-dot-dot: tables only
        (True, False, True): (0, (3, 1, 1, 1, 1, 1)),  # Complex dash: reranking + tables
        (False, True, True): (0, (1, 1)),  # Densely dotted: hybrid + tables
        (True, True, True): (0, (5, 1))   # Loosely dashed: all features
    }
    
    # Define markers for chunk sizes
    size_markers = {
        '100': 'o',      # circle
        '200': 's',      # square
        '300': '^',      # triangle up
        '400': 'd',      # diamond
        '500': 'p',      # pentagon
        '1000': '*'      # star
    }
    
    # Plot consecutive recall@k for all configurations
    for config_name, results in all_results.items():
        # Parse configuration name to extract components
        chunking_match = re.search(r'(token|semantic|late)', config_name)
        chunking = chunking_match.group(1) if chunking_match else "unknown"
        
        embedding_match = re.search(r'(bge-small-en-v1\.5|bge-large-en-v1\.5|bge-base-en-v1\.5|e5-small-v2)', config_name)
        embedding = embedding_match.group(1) if embedding_match else "unknown"
        
        size_match = re.search(r'chunk(\d+)', config_name)
        size = size_match.group(1) if size_match else "0"
        
        # Determine variations
        reranking = 'with_reranking' in config_name
        hybrid = 'hybrid' in config_name
        tables = 'with_tables' in config_name
        
        # Get base configuration
        base_config = f"{chunking}_{size}_{embedding}"
        
        # Get color from base configuration
        color = base_config_colors.get(base_config, (0.5, 0.5, 0.5, 1.0))  # Default to gray if not found
        
        # Get line style from variations
        linestyle = variation_styles.get((reranking, hybrid, tables), '-')
        
        # Get marker from size
        marker = size_markers.get(size, 'o')
        
        # Create a clear, descriptive label
        embedding_short = embedding.split('-')[0]  # Just use 'bge' or 'e5'
        
        # Format the label with all relevant information
        label_parts = [f"{chunking}-{size}"]
        
        # Add embedding model info
        if embedding_short:
            label_parts.append(embedding_short)
        
        # Add variation info
        variations = []
        if reranking:
            variations.append("rerank")
        if hybrid:
            variations.append("hybrid")
        if tables:
            variations.append("tables")
        
        if variations:
            label_parts.append("+".join(variations))
        
        label = " | ".join(label_parts)
        
        # Get data - handle missing keys gracefully
        recall_values = []
        for k in k_values:
            try:
                if "consecutive_recall_at_k" in results and str(k) in results["consecutive_recall_at_k"]:
                    recall_values.append(results["consecutive_recall_at_k"][str(k)])
                else:
                    recall_values.append(0)
            except (KeyError, TypeError):
                recall_values.append(0)
        
        # Only plot if we have valid data
        if any(v > 0 for v in recall_values):
            # Plot with appropriate style
            plt.plot(k_values, recall_values, 
                    marker=marker, linestyle=linestyle, color=color, 
                    linewidth=2, label=label,
                    markersize=8)
    
    # Add labels and title
    plt.xlabel('k (Number of Retrieved Documents)', fontsize=14)
    plt.ylabel('Consecutive Recall@k', fontsize=14)
    plt.title('Consecutive Recall@k for Different RAG Configurations', fontsize=16)
    
    # Use linear scale for x-axis
    plt.xscale('linear')
    
    # Set x-axis ticks to match k_values
    plt.xticks(k_values)
    
    # Set y-axis limits to start from 0
    plt.ylim(bottom=0)
    
    # Improve grid appearance
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend with better placement and formatting
    # Use ncol parameter to create multiple columns in the legend
    num_configs = len([c for c in all_results.keys() if any(v > 0 for v in [results["consecutive_recall_at_k"].get(str(k), 0) for k in k_values] if "consecutive_recall_at_k" in results)])
    ncols = min(3, max(1, num_configs // 6))  # Adjust number of columns based on number of configs
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=ncols, fontsize=10, frameon=True, facecolor='white', edgecolor='gray',
              shadow=True)
    
    # Adjust layout to make the plot take up more space
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Leave space for the legend at the bottom
    
    # Add a grid to make it easier to read values
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save plot with high resolution
    plt.savefig(os.path.join(output_dir, "comparison_recall_at_k.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a similar plot for MRR@k if available
    if any("mrr_at_k" in results for results in all_results.values()):
        plt.figure(figsize=(16, 10))
        
        for config_name, results in all_results.items():
            if "mrr_at_k" not in results:
                continue
                
            # Parse configuration name to extract components (same as above)
            chunking_match = re.search(r'(token|semantic|late)', config_name)
            chunking = chunking_match.group(1) if chunking_match else "unknown"
            
            embedding_match = re.search(r'(bge-small-en-v1\.5|bge-large-en-v1\.5|bge-base-en-v1\.5|e5-small-v2)', config_name)
            embedding = embedding_match.group(1) if embedding_match else "unknown"
            
            size_match = re.search(r'chunk(\d+)', config_name)
            size = size_match.group(1) if size_match else "0"
            
            # Determine variations
            reranking = 'with_reranking' in config_name
            hybrid = 'hybrid' in config_name
            tables = 'with_tables' in config_name
            
            # Get base configuration
            base_config = f"{chunking}_{size}_{embedding}"
            
            # Get color from base configuration
            color = base_config_colors.get(base_config, (0.5, 0.5, 0.5, 1.0))
            
            # Get line style from variations
            linestyle = variation_styles.get((reranking, hybrid, tables), '-')
            
            # Get marker from size
            marker = size_markers.get(size, 'o')
            
            # Create a clear, descriptive label
            embedding_short = embedding.split('-')[0]  # Just use 'bge' or 'e5'
            
            # Format the label with all relevant information
            label_parts = [f"{chunking}-{size}"]
            
            # Add embedding model info
            if embedding_short:
                label_parts.append(embedding_short)
            
            # Add variation info
            variations = []
            if reranking:
                variations.append("rerank")
            if hybrid:
                variations.append("hybrid")
            if tables:
                variations.append("tables")
            
            if variations:
                label_parts.append("+".join(variations))
            
            label = " | ".join(label_parts)
            
            # Get data - handle missing keys gracefully
            mrr_values = []
            for k in k_values:
                try:
                    if "mrr_at_k" in results and str(k) in results["mrr_at_k"]:
                        mrr_values.append(results["mrr_at_k"][str(k)])
                    else:
                        mrr_values.append(0)
                except (KeyError, TypeError):
                    mrr_values.append(0)
            
            # Only plot if we have valid data
            if any(v > 0 for v in mrr_values):
                # Plot with appropriate style
                plt.plot(k_values, mrr_values, 
                        marker=marker, linestyle=linestyle, color=color, 
                        linewidth=2, label=label,
                        markersize=8)
        
        # Add labels and title
        plt.xlabel('k (Number of Retrieved Documents)', fontsize=14)
        plt.ylabel('MRR@k', fontsize=14)
        plt.title('Mean Reciprocal Rank (MRR@k) for Different RAG Configurations', fontsize=16)
        
        # Use linear scale for x-axis
        plt.xscale('linear')
        
        # Set x-axis ticks to match k_values
        plt.xticks(k_values)
        
        # Set y-axis limits to start from 0
        plt.ylim(bottom=0)
        
        # Improve grid appearance
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Add legend with better placement and formatting
        num_configs = len([c for c in all_results.keys() if "mrr_at_k" in results and any(v > 0 for v in [results["mrr_at_k"].get(str(k), 0) for k in k_values])])
        ncols = min(3, max(1, num_configs // 6))
        
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=ncols, fontsize=10, frameon=True, facecolor='white', edgecolor='gray',
                  shadow=True)
        
        # Adjust layout to make the plot take up more space
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        
        # Add a grid to make it easier to read values
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save plot with high resolution
        plt.savefig(os.path.join(output_dir, "comparison_mrr_at_k.png"), dpi=300, bbox_inches='tight')
        plt.close()