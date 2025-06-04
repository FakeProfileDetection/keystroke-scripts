"""
ml_platforms_visualizer.py - Visualization and reporting functionality.
"""

from pathlib import Path
from typing import List, Tuple, Any
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from catboost import Pool
from xgboost import plot_importance
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Visualizer:
    """Handles all visualization and reporting functionality."""
    
    def __init__(self, config, output_dir: Path, timestamp: str):
        self.config = config
        self.output_dir = output_dir
        self.timestamp = timestamp
    
    def plot_platform_leakage(self, features: List[Tuple[str, float]]):
        """Plot platform leakage analysis."""
        plt.figure(figsize=(12, 8))
        names, scores = zip(*features)
        colors = ['red' if score > 0.1 else 'orange' if score > 0.05 else 'green' for score in scores]
        
        feature_names = [name[:30] + '...' if len(name) > 30 else name for name in names]
        
        plt.barh(range(len(feature_names)), scores, color=colors)
        plt.yticks(range(len(feature_names)), feature_names)
        plt.xlabel("Feature Importance")
        plt.title("Top Platform-Leaking Features\nRed: High Risk | Orange: Moderate Risk | Green: Low Risk")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(self.output_dir / "platform_leakage_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_confusion_matrices(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                  title: str, filename: str, k_values=[1, 5]):
        """Create confusion matrices for top-k predictions."""
        for k in k_values:
            if k > y_pred_proba.shape[1]:
                continue
            
            if k == 1:
                # Top-1 confusion matrix
                y_pred_top1 = np.argmax(y_pred_proba, axis=1)
                cm = confusion_matrix(y_true, y_pred_top1)
                
                # Get unique labels
                unique_labels = sorted(list(set(y_true) | set(y_pred_top1)))
                n_users = len(unique_labels)
                fig_width = max(12, min(50, n_users * 0.8))
                fig_height = max(10, min(40, n_users * 0.6))
                
                plt.figure(figsize=(fig_width, fig_height))
                
                if len(unique_labels) <= 50:
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=unique_labels, yticklabels=unique_labels,
                               cbar_kws={'shrink': 0.8})
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=0)
                else:
                    x_labels = [label if i % 2 == 0 else '' for i, label in enumerate(unique_labels)]
                    y_labels = [label if i % 2 == 0 else '' for i, label in enumerate(unique_labels)]
                    
                    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                               xticklabels=x_labels, yticklabels=y_labels,
                               cbar_kws={'shrink': 0.6})
                    plt.xticks(rotation=90, ha='center', fontsize=8)
                    plt.yticks(rotation=0, fontsize=8)
                
                plt.title(f'Top-1 Confusion Matrix - {title}')
                plt.ylabel('True User ID')
                plt.xlabel('Predicted User ID')
                plt.tight_layout()
                plt.savefig(self.output_dir / f'{filename}_top_1_confusion_matrix.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            else:
                # Top-k identification success matrix
                top_k_indices = np.argsort(y_pred_proba, axis=1)[:, -k:]
                top_k_correct = np.array([true_label in top_k_pred for true_label, top_k_pred in zip(y_true, top_k_indices)])
                
                binary_true = np.ones_like(top_k_correct)
                binary_pred = top_k_correct.astype(int)
                cm_binary = confusion_matrix(binary_true, binary_pred, labels=[0, 1])
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=[f'Not in Top-{k}', f'In Top-{k}'], 
                           yticklabels=['User Not Found', 'User Found'])
                plt.title(f'Top-{k} Identification Success - {title}')
                plt.ylabel('Expected Outcome')
                plt.xlabel('Actual Outcome')
                plt.tight_layout()
                plt.savefig(self.output_dir / f'{filename}_top_{k}_identification_matrix.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    def plot_feature_importance(self, model: Any, model_name: str, experiment_name: str, 
                               feature_names: List[str]):
        """Plot feature importance for tree-based models."""
        if not self.config.draw_feature_importance:
            return
        
        try:
            plt.figure(figsize=(12, 8))
            
            if model_name == "XGBoost":
                plot_importance(model, importance_type="weight", max_num_features=15)
                plt.title(f"XGBoost Feature Importance - {experiment_name}")
            
            elif model_name in ["RandomForest", "ExtraTrees", "GradientBoosting"]:
                importances = model.feature_importances_
                indices = np.argsort(importances)[-15:]
                plt.barh(range(len(indices)), importances[indices], align="center")
                plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
                plt.xlabel("Feature Importance")
                plt.title(f"Random Forest Feature Importance - {experiment_name}")
                
            elif model_name == "LightGBM":
                # LightGBM has feature_importances_ like sklearn models
                importances = model.feature_importances_
                indices = np.argsort(importances)[-15:]
                plt.barh(range(len(indices)), importances[indices], align="center")
                plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
                plt.xlabel("Feature Importance")
                plt.title(f"LightGBM Feature Importance - {experiment_name}")
            
            elif model_name == "CatBoost":
                # Create a dummy pool for feature importance
                dummy_X = np.random.randn(10, len(feature_names))
                dummy_y = np.random.randint(0, 2, 10)
                pool = Pool(dummy_X, label=dummy_y)
                
                importances = model.get_feature_importance(pool)
                sorted_indices = importances.argsort()[-15:]
                plt.barh(range(len(sorted_indices)), importances[sorted_indices], align="center")
                plt.yticks(range(len(sorted_indices)), [feature_names[i] for i in sorted_indices])
                plt.xlabel("Feature Importance")
                plt.title(f"CatBoost Feature Importance - {experiment_name}")
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{model_name.lower()}_feature_importance_{experiment_name}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️ Could not generate feature importance plot for {model_name}: {e}")
    
    def create_performance_plots(self, results_df: pd.DataFrame):
        """Create comprehensive performance plots with Plotly."""
        if results_df.empty:
            print("⚠️ No results to plot")
            return
        
        # Remove seed from experiment experiment name if it exists
        results_df['experiment'] = results_df["experiment"].str.replace(r'_seed\d+','',regex=True)
    
        # Create comprehensive performance plots
        fig = make_subplots(
            rows=2, cols=2,
            # subplot_titles=('Top-1 Accuracy by Model & Experiment', 'Top-K Accuracy Trends', 
            #                'F1 Score vs Top-1 Accuracy', 'Model Performance Overview'),
            subplot_titles=('Top-1 Accuracy by Model & Experiment', 'Top-K Accuracy Trends', 
                        'Top-5 Accuracy by Model & Experiment', 'Model Performance Overview'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        models = results_df['model'].unique()
        colors = px.colors.qualitative.Set1[:len(models)]
        
        # 1. Top-1 accuracy by model (by experiment)
        for i, model in enumerate(models):
            model_data = results_df[results_df['model'] == model]
            model_data = model_data.sort_values(by="experiment")
            
            if 'test_top_1_accuracy' in model_data.columns:
                fig.add_trace(
                    go.Scatter(x=model_data['experiment'], y=model_data['test_top_1_accuracy'],
                            name=f'{model}', line=dict(color=colors[i]),
                            mode='markers'),
                    row=1, col=1
                )

                # Calculate means for each experiment
                means_data = model_data.groupby('experiment')['test_top_1_accuracy'].mean().reset_index()
                
                # Plot means with a line (no markers)
                fig.add_trace(
                    go.Scatter(x=means_data['experiment'], y=means_data['test_top_1_accuracy'],
                            name=f'{model} (mean)', line=dict(color=colors[i], dash='dash'),
                            mode='lines', showlegend=False),
                    row=1, col=1
                )
        
        # 2. Top-K accuracy trends
        k_values = [1, 2, 3, 4, 5]
        for i, model in enumerate(models):
            model_data = results_df[results_df['model'] == model]
            k_accuracies = []
            for k in k_values:
                col_name = f'test_top_{k}_accuracy'
                if col_name in model_data.columns:
                    avg_acc = model_data[col_name].fillna(0).mean()
                    k_accuracies.append(avg_acc)
                else:
                    k_accuracies.append(0)
            
            fig.add_trace(
                go.Scatter(x=k_values, y=k_accuracies,
                        name=f'{model}', line=dict(color=colors[i]),
                        mode='lines+markers', showlegend=False),
                row=1, col=2
            )
        
        # 3. F1 vs Top-1 Accuracy scatter plot
        # for i, model in enumerate(models):
        #     model_data = results_df[results_df['model'] == model]
        #     if 'test_top_1_accuracy' in model_data.columns and 'test_f1_weighted' in model_data.columns:
        #         fig.add_trace(
        #             go.Scatter(x=model_data['test_f1_weighted'].fillna(0), 
        #                       y=model_data['test_top_1_accuracy'].fillna(0),
        #                       name=f'{model}', mode='markers', 
        #                       marker=dict(color=colors[i], size=8),
        #                       showlegend=False),
        #             row=2, col=1
        #         )

        # 3. Experiment versus Top-5 Accuracy
        for i, model in enumerate(models):
            model_data = results_df[results_df['model'] == model]
            model_data = model_data.sort_values(by="experiment")
            
            if 'test_top_5_accuracy' in model_data.columns:
                fig.add_trace(
                    go.Scatter(x=model_data['experiment'], y=model_data['test_top_5_accuracy'],
                            name=f'{model}', line=dict(color=colors[i]),
                            mode='markers',showlegend=False),
                    row=2, col=1
                )

                # Calculate means for each experiment
                means_data = model_data.groupby('experiment')['test_top_5_accuracy'].mean().reset_index()
                
                # Plot means with a line (no markers)
                fig.add_trace(
                    go.Scatter(x=means_data['experiment'], y=means_data['test_top_5_accuracy'],
                            name=f'{model} (mean)', line=dict(color=colors[i], dash='dash',),
                            mode='lines'),
                    row=2, col=1
                )
        
        # 4. Model Performance Overview
        model_names = []
        avg_top1_scores = []
        avg_top5_scores = []
        
        for model in models:
            model_data = results_df[results_df['model'] == model]
            avg_top1 = model_data.get('test_top_1_accuracy', pd.Series([0])).fillna(0).mean()
            avg_top5 = model_data.get('test_top_5_accuracy', pd.Series([0])).fillna(0).mean()
            
            model_names.append(model)
            avg_top1_scores.append(avg_top1)
            avg_top5_scores.append(avg_top5)
        
        # Add bars
        fig.add_trace(
            go.Bar(x=model_names, y=avg_top1_scores, 
                name='Avg Top-1', marker_color='teal'),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Bar(x=model_names, y=avg_top5_scores, 
                name='Avg Top-5', marker_color='purple'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800, 
            title_text="User Identification Performance Analysis",
            barmode='group'
        )
        
        # Add axis labels
        fig.update_xaxes(title_text="Experiment", row=1, col=1)
        fig.update_yaxes(title_text="Top-1 Accuracy", row=1, col=1)
        fig.update_xaxes(title_text="K Value", row=1, col=2)
        fig.update_yaxes(title_text="Top-K Accuracy", row=1, col=2)
        fig.update_xaxes(title_text="Experiment", row=2, col=1)
        fig.update_yaxes(title_text="Top-5 Accuracy", row=2, col=1)
        fig.update_xaxes(title_text="Model", row=2, col=2)
        fig.update_yaxes(title_text="Average Accuracy", row=2, col=2)
        
        fig.write_html(self.output_dir / "performance_plots.html")
        print(f"📈 Performance plots saved to: {self.output_dir / 'performance_plots.html'}")
    
    def generate_comprehensive_html_report(self, results_df: pd.DataFrame, detailed_df: pd.DataFrame):
        """Generate comprehensive HTML report."""
        if results_df.empty:
            print("⚠️ No experiment results to report")
            return
        
        results_df = results_df.sort_values(by="experiment")
        
        # Find best performers
        best_top1_idx = results_df['test_top_1_accuracy'].fillna(0).idxmax()
        best_top5_idx = results_df['test_top_5_accuracy'].fillna(0).idxmax()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>User Identification Results - {self.timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 10px; margin-bottom: 30px; }}
                .section {{ background-color: white; padding: 20px; margin-bottom: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #34495e; color: white; }}
                .metric-highlight {{ font-weight: bold; color: #27ae60; }}
                .info-box {{ background-color: #ecf0f1; padding: 15px; border-left: 4px solid #3498db; margin-bottom: 20px; }}
                .debug-mode {{ background-color: #ffe6e6; color: #d63031; padding: 10px; border-radius: 5px; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
        """
        
        # Add header section
        html_content += f"""
            <div class="header">
                <h1>User Identification Analysis</h1>
                <p><strong>Timestamp:</strong> {self.timestamp}</p>
                <p><strong>Task:</strong> Cross-Platform User Identification</p>
                <p><strong>Early Stopping:</strong> {'Enabled' if self.config.early_stopping else 'Disabled'}</p>
                <p><strong>Debug Mode:</strong> {'Enabled' if self.config.debug_mode else 'Disabled'}</p>
                <p><strong>Total Model Runs:</strong> {len(results_df)}</p>
                <p><strong>Data:</strong> Pre-normalized (no additional scaling)</p>
            </div>
            
            {'<div class="debug-mode"><strong>⚠️ DEBUG MODE:</strong> Results generated with minimal hyperparameter search for testing purposes only.</div>' if self.config.debug_mode else ''}
            
            <div class="section">
                <h2>Best Performance Summary</h2>
                <div class="info-box">
                    <p><strong>Best Top-1 Accuracy:</strong> <span class="metric-highlight">{results_df.loc[best_top1_idx, 'test_top_1_accuracy']:.4f}</span></p>
                    <p><strong>Best Top-1 Model:</strong> <span class="metric-highlight">{results_df.loc[best_top1_idx, 'model']} ({results_df.loc[best_top1_idx, 'experiment']})</span></p>
                    <p><strong>Best Top-5 Accuracy:</strong> <span class="metric-highlight">{results_df.loc[best_top5_idx, 'test_top_5_accuracy']:.4f}</span></p>
                    <p><strong>Best Top-5 Model:</strong> <span class="metric-highlight">{results_df.loc[best_top5_idx, 'model']} ({results_df.loc[best_top5_idx, 'experiment']})</span></p>
                </div>
            </div>
        """
        
        # Add model performance breakdown
        html_content += """
            
            <div class="section">
                <h2>Model Performance Breakdown</h2>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Avg Top-1</th>
                        <th>Avg Top-5</th>
                        <th>Best Experiment</th>
                        <th>Std Dev</th>
                    </tr>
        """
        for model in results_df['model'].unique():
            model_data = results_df[results_df['model'] == model]
            avg_top1 = model_data['test_top_1_accuracy'].mean()
            avg_top5 = model_data['test_top_5_accuracy'].mean()
            std_top1 = model_data['test_top_1_accuracy'].std()
            best_exp = model_data.loc[model_data['test_top_1_accuracy'].idxmax(), 'experiment']
            
            html_content += f"""
                    <tr>
                        <td>{model}</td>
                        <td>{avg_top1:.4f}</td>
                        <td>{avg_top5:.4f}</td>
                        <td>{best_exp}</td>
                        <td>{std_top1:.4f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
            """
        
        # Add top-k performance table
        html_content += """
            <div class="section">
                <h2>Mean Top-K Performance Summary</h2>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Experiment</th>
                        <th>Top-1</th>
                        <th>Top-2</th>
                        <th>Top-3</th>
                        <th>Top-4</th>
                        <th>Top-5</th>
                    </tr>
        """
        
        # Get statistics for each model/experiment combination
        for (model, experiment), group_df in results_df.groupby(['model', 'experiment']):
            stats = group_df.describe()
            
            html_content += f"""
                    <tr>
                        <td>{model}</td>
                        <td>{experiment}</td>
                        <td>{stats.loc['mean', 'test_top_1_accuracy']:.4f} ± {stats.loc['std', 'test_top_1_accuracy']:.4f}</td>
                        <td>{stats.loc['mean', 'test_top_2_accuracy']:.4f} ± {stats.loc['std', 'test_top_2_accuracy']:.4f}</td>
                        <td>{stats.loc['mean', 'test_top_3_accuracy']:.4f} ± {stats.loc['std', 'test_top_3_accuracy']:.4f}</td>
                        <td>{stats.loc['mean', 'test_top_4_accuracy']:.4f} ± {stats.loc['std', 'test_top_4_accuracy']:.4f}</td>
                        <td>{stats.loc['mean', 'test_top_5_accuracy']:.4f} ± {stats.loc['std', 'test_top_5_accuracy']:.4f}</td>
                    </tr>
            """

        html_content += f"""
                </table>
            </div>
            """
            
        # Get max and min values
        html_content += """
            <div class="section">
                <h2>Max/min Top-K Performance Summary</h2>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Experiment</th>
                        <th>Top-1</th>
                        <th>Top-2</th>
                        <th>Top-3</th>
                        <th>Top-4</th>
                        <th>Top-5</th>
                    </tr>
        """
        
        # Get statistics for each model/experiment combination
        for (model, experiment), group_df in results_df.groupby(['model', 'experiment']):
            stats = group_df.describe()
            
            html_content += f"""
                    <tr>
                        <td>{model}</td>
                        <td>{experiment}</td>
                        <td>{stats.loc['max', 'test_top_1_accuracy']:.4f} - {stats.loc['min', 'test_top_1_accuracy']:.4f}</td>
                        <td>{stats.loc['max', 'test_top_2_accuracy']:.4f} - {stats.loc['min', 'test_top_2_accuracy']:.4f}</td>
                        <td>{stats.loc['max', 'test_top_3_accuracy']:.4f} - {stats.loc['min', 'test_top_3_accuracy']:.4f}</td>
                        <td>{stats.loc['max', 'test_top_4_accuracy']:.4f} - {stats.loc['min', 'test_top_4_accuracy']:.4f}</td>
                        <td>{stats.loc['max', 'test_top_5_accuracy']:.4f} - {stats.loc['min', 'test_top_5_accuracy']:.4f}</td>
                    </tr>
            """

        html_content += f"""
                </table>
            </div>
            """
        
        
        # Add Generated Files section
        html_content += """
            <div class="section">
                <h2>Generated Files</h2>
                <ul>
                    <li><strong>Summary Results:</strong> experiment_results_{self.timestamp}.csv</li>
                    <li><strong>Detailed Top-K Results:</strong> detailed_topk_results_{self.timestamp}.csv</li>
                    <li><strong>Performance Plots:</strong> performance_plots.html</li>
                    <li><strong>Trained Models:</strong> {len([r for r in results_df.iterrows() if 'model_path' in r[1]])} models with metadata</li>
                    <li><strong>Confusion Matrices:</strong> Enhanced Top-1 and Top-5 visualizations for each model</li>
                    <li><strong>Feature Importance:</strong> Available for tree-based models (if enabled)</li>
                </ul>
            </div>
            """
            
        # End
        html_content += """
        </body>
        </html>
        """
        
        # Save HTML report
        html_path = self.output_dir / f"user_identification_report.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📋 HTML report: {html_path}")
        return html_path
