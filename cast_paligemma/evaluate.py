"""
Evaluation utilities for PaliGemma CAST model.
Additional metrics and visualization tools.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Tuple


class CASTEvaluator:
    """Comprehensive evaluator for CAST model predictions."""
    
    def __init__(self, results_csv: str):
        """Initialize with results CSV file."""
        self.df = pd.read_csv(results_csv)
        self.results_dir = Path(results_csv).parent / "evaluation"
        self.results_dir.mkdir(exist_ok=True)
        
    def compute_detailed_metrics(self) -> Dict[str, float]:
        """Compute detailed evaluation metrics."""
        # Extract predictions and targets
        pred_cols = [f'pred_{axis}_{step}' for step in range(8) for axis in ['x', 'y']]
        target_cols = [f'target_{axis}_{step}' for step in range(8) for axis in ['x', 'y']]
        
        predictions = self.df[pred_cols].values.reshape(-1, 8, 2)
        targets = self.df[target_cols].values.reshape(-1, 8, 2)
        
        metrics = {}
        
        # Overall metrics
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(mse)
        
        # Per-axis metrics
        mse_x = np.mean((predictions[:, :, 0] - targets[:, :, 0]) ** 2)
        mse_y = np.mean((predictions[:, :, 1] - targets[:, :, 1]) ** 2)
        mae_x = np.mean(np.abs(predictions[:, :, 0] - targets[:, :, 0]))
        mae_y = np.mean(np.abs(predictions[:, :, 1] - targets[:, :, 1]))
        
        # Per-step metrics
        step_mse = np.mean((predictions - targets) ** 2, axis=(0, 2))
        step_mae = np.mean(np.abs(predictions - targets), axis=(0, 2))
        
        # Distance-based metrics
        distances = np.linalg.norm(predictions - targets, axis=2)  # Shape: (N, 8)
        mean_distance = np.mean(distances)
        final_distance = np.mean(distances[:, -1])  # Final step distance
        
        # Success rates at different thresholds
        thresholds = [0.05, 0.1, 0.2, 0.5]
        success_rates = {}
        for thresh in thresholds:
            success_rates[f'success_rate_{thresh}'] = np.mean(distances < thresh)
        
        # Trajectory-level metrics
        cumulative_distances = np.sum(distances, axis=1)
        mean_cumulative_distance = np.mean(cumulative_distances)
        
        # Compile all metrics
        metrics.update({
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mse_x': mse_x,
            'mse_y': mse_y,
            'mae_x': mae_x,
            'mae_y': mae_y,
            'mean_distance': mean_distance,
            'final_distance': final_distance,
            'mean_cumulative_distance': mean_cumulative_distance,
            **success_rates
        })
        
        # Add per-step metrics
        for step in range(8):
            metrics[f'step_{step}_mse'] = step_mse[step]
            metrics[f'step_{step}_mae'] = step_mae[step]
            
        return metrics
    
    def plot_error_distributions(self):
        """Plot error distributions."""
        pred_cols = [f'pred_{axis}_{step}' for step in range(8) for axis in ['x', 'y']]
        target_cols = [f'target_{axis}_{step}' for step in range(8) for axis in ['x', 'y']]
        
        predictions = self.df[pred_cols].values.reshape(-1, 8, 2)
        targets = self.df[target_cols].values.reshape(-1, 8, 2)
        
        errors = predictions - targets
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # X-axis errors
        axes[0, 0].hist(errors[:, :, 0].flatten(), bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title('X-axis Error Distribution')
        axes[0, 0].set_xlabel('Error')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(0, color='red', linestyle='--')
        
        # Y-axis errors  
        axes[0, 1].hist(errors[:, :, 1].flatten(), bins=50, alpha=0.7, color='green')
        axes[0, 1].set_title('Y-axis Error Distribution')
        axes[0, 1].set_xlabel('Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(0, color='red', linestyle='--')
        
        # Distance errors
        distances = np.linalg.norm(errors, axis=2)
        axes[1, 0].hist(distances.flatten(), bins=50, alpha=0.7, color='purple')
        axes[1, 0].set_title('Distance Error Distribution')
        axes[1, 0].set_xlabel('Distance Error')
        axes[1, 0].set_ylabel('Frequency')
        
        # Error by step
        step_errors = np.mean(distances, axis=0)
        axes[1, 1].bar(range(8), step_errors, color='orange')
        axes[1, 1].set_title('Mean Distance Error by Step')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Mean Distance Error')
        axes[1, 1].set_xticks(range(8))
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "error_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_trajectory_examples(self, num_examples: int = 6):
        """Plot example trajectories comparing predictions vs targets."""
        pred_cols = [f'pred_{axis}_{step}' for step in range(8) for axis in ['x', 'y']]
        target_cols = [f'target_{axis}_{step}' for step in range(8) for axis in ['x', 'y']]
        
        predictions = self.df[pred_cols].values.reshape(-1, 8, 2)
        targets = self.df[target_cols].values.reshape(-1, 8, 2)
        
        # Select random examples
        indices = np.random.choice(len(predictions), num_examples, replace=False)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, idx in enumerate(indices):
            ax = axes[i]
            
            # Plot trajectories
            ax.plot(predictions[idx, :, 0], predictions[idx, :, 1], 'b-o', label='Predicted', markersize=4)
            ax.plot(targets[idx, :, 0], targets[idx, :, 1], 'r-s', label='Ground Truth', markersize=4)
            
            # Mark start and end
            ax.plot(predictions[idx, 0, 0], predictions[idx, 0, 1], 'go', markersize=8, label='Start')
            ax.plot(predictions[idx, -1, 0], predictions[idx, -1, 1], 'mo', markersize=8, label='End')
            
            # Add step numbers
            for step in range(8):
                ax.annotate(f'{step}', 
                           (predictions[idx, step, 0], predictions[idx, step, 1]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax.set_title(f'Sample {idx}\n"{self.df.iloc[idx]["instruction"][:30]}..."')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.axis('equal')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "trajectory_examples.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_instruction_performance(self):
        """Analyze performance by instruction characteristics."""
        # Compute distance error for each sample
        pred_cols = [f'pred_{axis}_{step}' for step in range(8) for axis in ['x', 'y']]
        target_cols = [f'target_{axis}_{step}' for step in range(8) for axis in ['x', 'y']]
        
        predictions = self.df[pred_cols].values.reshape(-1, 8, 2)
        targets = self.df[target_cols].values.reshape(-1, 8, 2)
        
        distances = np.linalg.norm(predictions - targets, axis=2)
        mean_distances = np.mean(distances, axis=1)
        
        # Add to dataframe
        analysis_df = self.df.copy()
        analysis_df['mean_distance_error'] = mean_distances
        analysis_df['instruction_length'] = analysis_df['instruction'].str.len()
        analysis_df['num_words'] = analysis_df['instruction'].str.split().str.len()
        
        # Performance by instruction length
        length_bins = pd.cut(analysis_df['instruction_length'], bins=5)
        length_performance = analysis_df.groupby(length_bins)['mean_distance_error'].agg(['mean', 'std', 'count'])
        
        # Performance by number of words
        word_bins = pd.cut(analysis_df['num_words'], bins=5)
        word_performance = analysis_df.groupby(word_bins)['mean_distance_error'].agg(['mean', 'std', 'count'])
        
        # Save analysis
        with open(self.results_dir / "instruction_analysis.txt", 'w') as f:
            f.write("Performance by Instruction Length:\n")
            f.write(str(length_performance))
            f.write("\n\nPerformance by Number of Words:\n")
            f.write(str(word_performance))
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Length analysis
        length_centers = [interval.mid for interval in length_performance.index]
        ax1.errorbar(length_centers, length_performance['mean'], 
                    yerr=length_performance['std'], marker='o', capsize=5)
        ax1.set_xlabel('Instruction Length (characters)')
        ax1.set_ylabel('Mean Distance Error')
        ax1.set_title('Performance vs Instruction Length')
        ax1.grid(True, alpha=0.3)
        
        # Word count analysis
        word_centers = [interval.mid for interval in word_performance.index]
        ax2.errorbar(word_centers, word_performance['mean'],
                    yerr=word_performance['std'], marker='s', capsize=5)
        ax2.set_xlabel('Number of Words')
        ax2.set_ylabel('Mean Distance Error')
        ax2.set_title('Performance vs Number of Words')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "instruction_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return analysis_df
    
    def generate_report(self):
        """Generate comprehensive evaluation report."""
        print("Generating evaluation report...")
        
        # Compute metrics
        metrics = self.compute_detailed_metrics()
        
        # Create visualizations
        self.plot_error_distributions()
        self.plot_trajectory_examples()
        analysis_df = self.analyze_instruction_performance()
        
        # Write report
        report_path = self.results_dir / "evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write("PaliGemma CAST Model Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Overall Metrics:\n")
            f.write("-" * 20 + "\n")
            for metric, value in metrics.items():
                if not metric.startswith('step_'):
                    f.write(f"{metric}: {value:.4f}\n")
            
            f.write(f"\nPer-Step Metrics:\n")
            f.write("-" * 20 + "\n")
            for step in range(8):
                mse = metrics[f'step_{step}_mse']
                mae = metrics[f'step_{step}_mae']
                f.write(f"Step {step}: MSE={mse:.4f}, MAE={mae:.4f}\n")
            
            f.write(f"\nDataset Statistics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total samples: {len(self.df)}\n")
            f.write(f"Mean instruction length: {analysis_df['instruction_length'].mean():.1f} characters\n")
            f.write(f"Mean number of words: {analysis_df['num_words'].mean():.1f} words\n")
            
            f.write(f"\nGenerated Files:\n")
            f.write("-" * 20 + "\n")
            f.write("- error_distributions.png: Error distribution plots\n")
            f.write("- trajectory_examples.png: Example trajectory visualizations\n")
            f.write("- instruction_performance.png: Performance vs instruction characteristics\n")
            f.write("- instruction_analysis.txt: Detailed instruction analysis\n")
        
        print(f"Evaluation report saved to: {report_path}")
        print(f"Visualizations saved to: {self.results_dir}")
        
        return metrics


def main():
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True, help="Path to results CSV file")
    args = parser.parse_args()
    
    evaluator = CASTEvaluator(args.results)
    metrics = evaluator.generate_report()
    
    print("\nEvaluation Summary:")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"Success Rate (0.1): {metrics['success_rate_0.1']:.4f}")


if __name__ == "__main__":
    main()