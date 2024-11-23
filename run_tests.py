import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from fabric_predictor import FabricPredictor
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def evaluate_model():
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('metric_reports', exist_ok=True)
    os.makedirs('metric_reports/plots', exist_ok=True)
    os.makedirs('metric_reports/data', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'metric_reports/data/test_results_{timestamp}.txt'
    
    # Initialize predictor and load models
    predictor = FabricPredictor()
    predictor.train_model()
    
    # Load and clean test data
    df = pd.read_csv('fabric_restructured.csv')
    df = predictor._clean_data(df)
    
    # Get actual predictions for each row
    y_true = []
    y_pred = []
    rule_based_correct = 0
    nn_correct = 0
    prediction_details = []
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("Fabric Predictor Model Evaluation Results (Rule-Based + Neural Network)\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        # Record predictions
        f.write("Individual Predictions:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Best Use':<20} {'Durability':<10} {'Texture':<10} {'Actual':<15} {'Predicted':<15} {'Match':<10} {'Rule-Based':<15} {'Neural Net':<15}\n")
        f.write("-"*110 + "\n")
        
        total_samples = len(df)
        correct_predictions = 0
        
        for _, row in df.iterrows():
            best_use = row['Best Use']
            durability = row['Durability']
            texture = row['Texture']
            true_fabric = row['Fabric Name']
            
            # Get rule-based prediction
            rule_probs, rule_fabrics = predictor.rule_based_model.predict_proba(best_use, durability, texture)
            rule_pred = rule_fabrics[np.argmax(rule_probs)]
            
            # Get neural network prediction
            # Create features for neural network
            X_best_use = predictor.le_best_use.transform([best_use])
            X_durability = predictor.le_durability.transform([durability])
            X_texture = predictor.le_texture.transform([texture])
            
            # Create all features as in predict method
            X_best_use_onehot = np.zeros((1, len(predictor.le_best_use.classes_)))
            X_best_use_onehot[0, X_best_use[0]] = 1
            
            X_durability_onehot = np.zeros((1, len(predictor.le_durability.classes_)))
            X_durability_onehot[0, X_durability[0]] = 1
            
            X_texture_onehot = np.zeros((1, len(predictor.le_texture.classes_)))
            X_texture_onehot[0, X_texture[0]] = 1
            
            interactions = predictor._create_interaction_features(X_best_use, X_durability, X_texture)
            
            X = np.column_stack([
                X_best_use,
                X_durability,
                X_texture,
                X_best_use_onehot,
                X_durability_onehot,
                X_texture_onehot,
                interactions
            ])
            
            X_poly = predictor._create_polynomial_features(X)
            X = np.column_stack([X, X_poly])
            X_scaled = predictor.scaler.transform(X)
            
            nn_pred_idx = np.argmax(predictor.nn_model.predict(X_scaled))
            nn_pred = predictor.le_fabric_name.inverse_transform([nn_pred_idx])[0]
            
            # Get final prediction
            pred_fabric, _ = predictor.predict(best_use, durability, texture)
            
            y_true.append(true_fabric)
            y_pred.append(pred_fabric)
            
            # Track individual model performance
            if rule_pred == true_fabric:
                rule_based_correct += 1
            if nn_pred == true_fabric:
                nn_correct += 1
            
            # Calculate prediction confidence
            is_correct = true_fabric == pred_fabric
            if is_correct:
                correct_predictions += 1
            
            current_accuracy = (correct_predictions / (len(y_true))) * 100
            
            # Write prediction details
            match_symbol = "[CORRECT]" if is_correct else "[WRONG]"
            rule_match = "[✓]" if rule_pred == true_fabric else "[✗]"
            nn_match = "[✓]" if nn_pred == true_fabric else "[✗]"
            
            f.write(f"{best_use:<20} {durability:<10} {texture:<10} {true_fabric:<15} {pred_fabric:<15} "
                   f"{match_symbol:<10} {rule_match:<15} {nn_match:<15}\n")
            
            prediction_details.append({
                'best_use': best_use,
                'durability': durability,
                'texture': texture,
                'true_fabric': true_fabric,
                'pred_fabric': pred_fabric,
                'rule_pred': rule_pred,
                'nn_pred': nn_pred,
                'is_correct': is_correct,
                'running_accuracy': current_accuracy
            })
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Write model performance comparison
        f.write("\nModel Performance Comparison:\n")
        f.write("-"*70 + "\n")
        f.write(f"Rule-Based Accuracy: {(rule_based_correct/total_samples)*100:.2f}%\n")
        f.write(f"Neural Network Accuracy: {(nn_correct/total_samples)*100:.2f}%\n")
        f.write(f"Ensemble Accuracy: {accuracy*100:.2f}%\n\n")
        
        # Write summary statistics
        f.write("\nSummary Statistics:\n")
        f.write("-"*70 + "\n")
        f.write(f"Total Samples: {total_samples}\n")
        f.write(f"Correct Predictions: {correct_predictions}\n")
        f.write(f"Incorrect Predictions: {total_samples - correct_predictions}\n")
        f.write(f"Final Accuracy: {accuracy*100:.2f}%\n\n")
        
        # Write detailed classification report
        f.write("\nDetailed Classification Report:\n")
        f.write("-"*70 + "\n")
        f.write(classification_report(y_true, y_pred))
        
        # Write overall metrics
        f.write("\nOverall Metrics:\n")
        f.write("-"*70 + "\n")
        f.write(f"Accuracy:  {accuracy:.3f}\n")
        f.write(f"Precision: {precision:.3f}\n")
        f.write(f"Recall:    {recall:.3f}\n")
        f.write(f"F1-Score:  {f1:.3f}\n")
        
        # Analysis by category
        f.write("\nPerformance Analysis by Category:\n")
        f.write("-"*70 + "\n")
        
        # Analyze by durability
        f.write("\nDurability-based Performance:\n")
        for durability in df['Durability'].unique():
            durability_mask = [d == durability for d in [p['durability'] for p in prediction_details]]
            if any(durability_mask):
                durability_acc = sum([1 for i, m in enumerate(durability_mask) if m and prediction_details[i]['is_correct']]) / sum(durability_mask)
                f.write(f"{durability}: {durability_acc*100:.2f}%\n")
        
        # Analyze by texture
        f.write("\nTexture-based Performance:\n")
        for texture in df['Texture'].unique():
            texture_mask = [t == texture for t in [p['texture'] for p in prediction_details]]
            if any(texture_mask):
                texture_acc = sum([1 for i, m in enumerate(texture_mask) if m and prediction_details[i]['is_correct']]) / sum(texture_mask)
                f.write(f"{texture}: {texture_acc*100:.2f}%\n")
        
        # Write per-class metrics
        f.write("\nPer-Class Performance:\n")
        f.write("-"*70 + "\n")
        unique_fabrics = sorted(set(y_true))
        for fabric in unique_fabrics:
            fabric_precision = precision_score(y_true, y_pred, labels=[fabric], average='weighted')
            fabric_recall = recall_score(y_true, y_pred, labels=[fabric], average='weighted')
            fabric_f1 = f1_score(y_true, y_pred, labels=[fabric], average='weighted')
            
            f.write(f"\n{fabric}:\n")
            f.write(f"  Precision: {fabric_precision:.3f}\n")
            f.write(f"  Recall:    {fabric_recall:.3f}\n")
            f.write(f"  F1-Score:  {fabric_f1:.3f}\n")
    
    # Create confusion matrix visualization
    plt.figure(figsize=(15, 10))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=sorted(set(y_true)),
                yticklabels=sorted(set(y_true)))
    plt.title('Confusion Matrix (Hybrid Model)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    
    # Save confusion matrix with timestamp
    plt.savefig(f'metric_reports/plots/confusion_matrix_{timestamp}.png', dpi=300, bbox_inches='tight')
    
    # Create accuracy progression plot
    plt.figure(figsize=(12, 6))
    accuracies = [p['running_accuracy'] for p in prediction_details]
    plt.plot(accuracies)
    plt.title('Accuracy Progression During Testing')
    plt.xlabel('Number of Predictions')
    plt.ylabel('Running Accuracy (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'metric_reports/plots/accuracy_progression_{timestamp}.png')
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    accuracies = {
        'Rule-Based': (rule_based_correct/total_samples)*100,
        'Neural Network': (nn_correct/total_samples)*100,
        'Ensemble': accuracy*100
    }
    plt.bar(accuracies.keys(), accuracies.values())
    plt.title('Model Performance Comparison')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    for i, v in enumerate(accuracies.values()):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center')
    plt.tight_layout()
    plt.savefig(f'metric_reports/plots/model_comparison_{timestamp}.png')
    
    print(f"Results have been saved to {results_file}")
    print(f"Plots have been saved to metric_reports/plots/")
    print(f"Models have been saved to models/")

if __name__ == "__main__":
    evaluate_model() 