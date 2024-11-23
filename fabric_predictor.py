import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import joblib
from typing import Tuple, Dict, List
import os

class RuleBasedClassifier:
    def __init__(self):
        self.rules = {}
        
        # Initialize expert rules in __init__
        self.expert_rules = {
            'winter': ['Wool', 'Fleece', 'Cashmere'],
            'summer': ['Linen', 'Cotton'],
            'evening': ['Silk', 'Satin', 'Chiffon'],
            'sportswear': ['Spandex', 'Polyester', 'Nylon'],
            'casual': ['Denim', 'Cotton'],
            'formal': ['Wool', 'Silk', 'Tweed'],
            'outdoor': ['Canvas', 'Nylon', 'Polyester'],
            'luxury': ['Cashmere', 'Silk', 'Velvet']
        }
        
        # Rules based on durability requirements
        self.durability_rules = {
            'High': ['Wool', 'Denim', 'Canvas', 'Leather', 'Polyester'],
            'Moderate': ['Cotton', 'Cashmere', 'Velvet'],
            'Low': ['Silk', 'Chiffon', 'Organza']
        }
        
        # Rules based on texture preferences
        self.texture_rules = {
            'Soft': ['Cashmere', 'Silk', 'Velvet', 'Fleece'],
            'Smooth': ['Cotton', 'Satin', 'Spandex'],
            'Rough': ['Wool', 'Tweed', 'Denim', 'Canvas']
        }
        
        # Add more specific category mappings
        self.fabric_categories = {
            'outerwear': {
                'winter': ['Wool', 'Fleece', 'Leather'],
                'summer': ['Linen', 'Cotton'],
                'casual': ['Denim', 'Canvas'],
                'formal': ['Tweed', 'Wool']
            },
            'dresses': {
                'evening': ['Silk', 'Satin', 'Chiffon'],
                'casual': ['Cotton', 'Rayon'],
                'party': ['Velvet', 'Silk'],
                'formal': ['Silk', 'Georgette']
            },
            'sportswear': {
                'active': ['Spandex', 'Nylon', 'Polyester'],
                'casual': ['Cotton', 'Polyester'],
                'professional': ['Polyester', 'Nylon']
            }
        }
        
        # Add texture-durability combinations
        self.texture_durability_rules = {
            ('Soft', 'High'): ['Fleece'],
            ('Soft', 'Moderate'): ['Cashmere', 'Velvet'],
            ('Soft', 'Low'): ['Silk', 'Chiffon'],
            ('Smooth', 'High'): ['Leather', 'Nylon'],
            ('Smooth', 'Moderate'): ['Cotton'],
            ('Smooth', 'Low'): ['Satin', 'Organza'],
            ('Rough', 'High'): ['Denim', 'Canvas', 'Tweed'],
            ('Rough', 'Moderate'): ['Wool'],
            ('Rough', 'Low'): ['Jute']
        }
        
        # Add seasonal rules
        self.seasonal_rules = {
            'winter': {
                'High': ['Wool', 'Fleece'],
                'Moderate': ['Cashmere', 'Velvet'],
                'Low': ['Silk']
            },
            'summer': {
                'High': ['Cotton', 'Polyester'],
                'Moderate': ['Linen', 'Rayon'],
                'Low': ['Chiffon', 'Georgette']
            }
        }
        
    def fit(self, df):
        # Create rules based on exact matches in training data
        for _, row in df.iterrows():
            key = (row['Best Use'], row['Durability'], row['Texture'])
            if key not in self.rules:
                self.rules[key] = []
            self.rules[key].append(row['Fabric Name'])
    
    def predict_proba(self, best_use, durability, texture):
        all_fabrics = list(set([fabric for fabrics in self.rules.values() for fabric in fabrics]))
        probs = np.zeros((1, len(all_fabrics)))
        
        # Check exact matches from training data
        key = (best_use, durability, texture)
        if key in self.rules:
            for fabric in self.rules[key]:
                idx = all_fabrics.index(fabric)
                probs[0, idx] += 0.4  # Weight for exact matches
        
        # Apply expert rules
        for category, fabrics in self.expert_rules.items():
            if category.lower() in best_use.lower():
                for fabric in fabrics:
                    if fabric in all_fabrics:
                        idx = all_fabrics.index(fabric)
                        probs[0, idx] += 0.2  # Weight for category matches
        
        # Apply durability rules
        for fabric in self.durability_rules[durability]:
            if fabric in all_fabrics:
                idx = all_fabrics.index(fabric)
                probs[0, idx] += 0.2  # Weight for durability matches
        
        # Apply texture rules
        for fabric in self.texture_rules[texture]:
            if fabric in all_fabrics:
                idx = all_fabrics.index(fabric)
                probs[0, idx] += 0.2  # Weight for texture matches
        
        # Add category-based predictions
        for category, subcategories in self.fabric_categories.items():
            if category.lower() in best_use.lower():
                for subcat, fabrics in subcategories.items():
                    if subcat.lower() in best_use.lower():
                        for fabric in fabrics:
                            if fabric in all_fabrics:
                                idx = all_fabrics.index(fabric)
                                probs[0, idx] += 0.3
        
        # Add texture-durability combination predictions
        if (texture, durability) in self.texture_durability_rules:
            for fabric in self.texture_durability_rules[(texture, durability)]:
                if fabric in all_fabrics:
                    idx = all_fabrics.index(fabric)
                    probs[0, idx] += 0.3
        
        # Add seasonal predictions
        for season, durability_dict in self.seasonal_rules.items():
            if season.lower() in best_use.lower():
                for dur, fabrics in durability_dict.items():
                    if dur == durability:
                        for fabric in fabrics:
                            if fabric in all_fabrics:
                                idx = all_fabrics.index(fabric)
                                probs[0, idx] += 0.2
        
        # Normalize probabilities
        row_sums = probs.sum(axis=1)
        probs = probs / row_sums[:, np.newaxis]
        
        return probs, all_fabrics

class FabricPredictor:
    def __init__(self):
        self.nn_model = None
        self.rule_based_model = RuleBasedClassifier()
        self.le_best_use = LabelEncoder()
        self.le_durability = LabelEncoder()
        self.le_texture = LabelEncoder()
        self.le_fabric_name = LabelEncoder()
        self.le_fabric_type = LabelEncoder()
        self.scaler = StandardScaler()
        
    def _clean_data(self, df):
        # Remove connecting words and clean Best Use column
        connecting_words = ['and', 'or', 'the', 'a', 'an']
        df = df[~df['Best Use'].str.lower().isin(connecting_words)]
        return df
    
    def _create_interaction_features(self, X_best_use, X_durability, X_texture):
        # Create interaction features
        interactions = np.column_stack([
            X_best_use * X_durability,
            X_best_use * X_texture,
            X_durability * X_texture,
            X_best_use * X_durability * X_texture
        ])
        return interactions
    
    def _create_polynomial_features(self, X):
        # Add polynomial features (squared terms)
        poly_features = np.column_stack([
            np.square(X),
            np.sqrt(np.abs(X))
        ])
        return poly_features
        
    def train_model(self):
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Load and clean the data
        df = pd.read_csv('fabric_restructured.csv')
        df = self._clean_data(df)
        
        # Train rule-based model
        self.rule_based_model.fit(df)
        
        # Encode features
        X_best_use = self.le_best_use.fit_transform(df['Best Use'])
        X_durability = self.le_durability.fit_transform(df['Durability'])
        X_texture = self.le_texture.fit_transform(df['Texture'])
        
        # Create additional features
        X_best_use_onehot = pd.get_dummies(df['Best Use'])
        X_durability_onehot = pd.get_dummies(df['Durability'])
        X_texture_onehot = pd.get_dummies(df['Texture'])
        
        # Create interaction features
        interactions = self._create_interaction_features(X_best_use, X_durability, X_texture)
        
        # Combine all features
        X = np.column_stack([
            X_best_use,
            X_durability,
            X_texture,
            X_best_use_onehot,
            X_durability_onehot,
            X_texture_onehot,
            interactions
        ])
        
        # Add polynomial features
        X_poly = self._create_polynomial_features(X)
        X = np.column_stack([X, X_poly])
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode targets
        y_fabric_name = self.le_fabric_name.fit_transform(df['Fabric Name'])
        y_fabric_type = self.le_fabric_type.fit_transform(df['Fabric Type'])
        
        # Compute class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_fabric_name),
            y=y_fabric_name
        )
        class_weight_dict = dict(zip(np.unique(y_fabric_name), class_weights))
        
        # Get dimensions for neural network
        input_dim = X_scaled.shape[1]
        num_classes = len(self.le_fabric_name.classes_)
        
        # Build enhanced neural network
        self.nn_model = Sequential([
            # Input layer
            Dense(512, activation='relu', input_dim=input_dim, kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layers with increasing then decreasing size
            Dense(1024, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(2048, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(1024, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.1),
            
            # Output layer
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile with optimized settings
        self.nn_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.0001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Enhanced callbacks
        callbacks = [
            EarlyStopping(
                monitor='loss',
                patience=50,
                restore_best_weights=True,
                min_delta=0.0001
            ),
            ReduceLROnPlateau(
                monitor='loss',
                factor=0.1,
                patience=20,
                min_lr=0.000001,
                min_delta=0.0001
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.keras',
                save_best_only=True,
                monitor='loss'
            )
        ]
        
        # Train with increased epochs and smaller batch size
        self.nn_model.fit(
            X_scaled,
            y_fabric_name,
            epochs=500,  # Increased epochs
            batch_size=32,   # Smaller batch size
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # Save models and preprocessors in models folder
        self.nn_model.save('models/nn_model.keras')
        joblib.dump(self.le_best_use, 'models/le_best_use.joblib')
        joblib.dump(self.le_durability, 'models/le_durability.joblib')
        joblib.dump(self.le_texture, 'models/le_texture.joblib')
        joblib.dump(self.le_fabric_name, 'models/le_fabric_name.joblib')
        joblib.dump(self.le_fabric_type, 'models/le_fabric_type.joblib')
        joblib.dump(self.scaler, 'models/scaler.joblib')
    
    def predict(self, best_use: str, durability: str, texture: str) -> Tuple[str, str]:
        # Get rule-based predictions
        rule_probs, rule_fabrics = self.rule_based_model.predict_proba(best_use, durability, texture)
        
        # Get neural network predictions
        X_best_use = self.le_best_use.transform([best_use])
        X_durability = self.le_durability.transform([durability])
        X_texture = self.le_texture.transform([texture])
        
        # Create features
        X_best_use_onehot = np.zeros((1, len(self.le_best_use.classes_)))
        X_best_use_onehot[0, X_best_use[0]] = 1
        
        X_durability_onehot = np.zeros((1, len(self.le_durability.classes_)))
        X_durability_onehot[0, X_durability[0]] = 1
        
        X_texture_onehot = np.zeros((1, len(self.le_texture.classes_)))
        X_texture_onehot[0, X_texture[0]] = 1
        
        # Create all features
        interactions = self._create_interaction_features(
            X_best_use, X_durability, X_texture
        )
        
        X = np.column_stack([
            X_best_use,
            X_durability,
            X_texture,
            X_best_use_onehot,
            X_durability_onehot,
            X_texture_onehot,
            interactions
        ])
        
        X_poly = self._create_polynomial_features(X)
        X = np.column_stack([X, X_poly])
        
        X_scaled = self.scaler.transform(X)
        
        # Get neural network predictions
        nn_pred_proba = self.nn_model.predict(X_scaled)
        
        # Convert rule-based predictions to match neural network fabric order
        rule_probs_aligned = np.zeros_like(nn_pred_proba)
        for i, fabric in enumerate(rule_fabrics):
            if fabric in self.le_fabric_name.classes_:
                idx = np.where(self.le_fabric_name.classes_ == fabric)[0][0]
                rule_probs_aligned[0, idx] = rule_probs[0, i]
        
        # Combine predictions (60% rule-based, 40% neural network)
        final_pred_proba = 0.6 * rule_probs_aligned + 0.4 * nn_pred_proba
        
        fabric_name_pred = np.argmax(final_pred_proba)
        fabric_name = self.le_fabric_name.inverse_transform([fabric_name_pred])[0]
        
        # Get fabric type
        df = pd.read_csv('fabric_restructured.csv')
        fabric_type = df[df['Fabric Name'] == fabric_name]['Fabric Type'].iloc[0]
        
        return fabric_name, fabric_type
    
    def load_model(self):
        self.nn_model = load_model('models/nn_model.keras')
        self.le_best_use = joblib.load('models/le_best_use.joblib')
        self.le_durability = joblib.load('models/le_durability.joblib')
        self.le_texture = joblib.load('models/le_texture.joblib')
        self.le_fabric_name = joblib.load('models/le_fabric_name.joblib')
        self.le_fabric_type = joblib.load('models/le_fabric_type.joblib')
        self.scaler = joblib.load('models/scaler.joblib')