#!/usr/bin/env python3
"""
NCAA Football Model Re-tuning Script
Phase 1: Data Update and Quick Wins
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ModelRetuner:
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load and prepare training data"""
        print("📊 Loading feature data...")
        try:
            # Load main feature dataset
            self.df = pd.read_csv("src/data/ncaa_games_last_15_years_features_with_elo.csv")
            print(f"✅ Loaded {len(self.df)} games from 15-year dataset")
            
            # Add recency weights (more recent games matter more)
            current_year = 2025
            self.df['recency_weight'] = np.exp((self.df['season'] - 2010) / 5)  # Exponential decay
            
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def prepare_features(self):
        """Prepare feature sets for different models"""
        print("🔧 Preparing feature sets...")
        
        # Core features that should be available
        self.base_features = [
            "week", "season",
            "home_conf_encoded", "away_conf_encoded", "venue_encoded",
            "home_elo", "away_elo"
        ]
        
        # Enhanced features (check availability)
        potential_features = [
            "home_total_points", "away_total_points", 
            "betting_spread", "betting_over_under",
            "home_season_avg_home", "away_season_avg_away",
            "home_recent_opp_avg_elo", "home_recent_opp_win_rate",
            "away_recent_opp_avg_elo", "away_recent_opp_win_rate"
        ]
        
        # Only include features that exist in the dataset
        available_features = [f for f in potential_features if f in self.df.columns]
        self.features = self.base_features + available_features
        
        print(f"✅ Using {len(self.features)} features: {self.features}")
        
        # Create feature matrix
        self.X = self.df[self.features]
        
        # Target variables
        self.y_total = self.df["game_total_points"] if "game_total_points" in self.df.columns else None
        self.y_margin = self.df["margin_of_victory"] if "margin_of_victory" in self.df.columns else None
        
        # Sample weights for recent games
        self.sample_weights = self.df['recency_weight']
        
    def create_train_test_split(self):
        """Create temporal train/test split"""
        print("📅 Creating temporal train/test split...")
        
        # Use 2024 as test set, everything before as training
        self.train_mask = self.df['season'] < 2024
        self.test_mask = self.df['season'] == 2024
        
        print(f"Training games: {self.train_mask.sum()}")
        print(f"Test games (2024): {self.test_mask.sum()}")
        
        # Split data
        self.X_train = self.X[self.train_mask]
        self.X_test = self.X[self.test_mask]
        self.weights_train = self.sample_weights[self.train_mask]
        
        if self.y_total is not None:
            self.y_total_train = self.y_total[self.train_mask]
            self.y_total_test = self.y_total[self.test_mask]
            
        if self.y_margin is not None:
            self.y_margin_train = self.y_margin[self.train_mask]
            self.y_margin_test = self.y_margin[self.test_mask]
    
    def tune_total_points_model(self):
        """Retune the total points prediction model"""
        if self.y_total is None:
            print("⚠️ No total points target variable found")
            return
            
        print("\n🎯 Tuning Total Points Model...")
        
        # Quick win: Better hyperparameters
        param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [15, 20, 25],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4]
        }
        
        # Grid search with cross-validation
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1
        )
        
        print("⚙️ Running hyperparameter optimization...")
        grid_search.fit(self.X_train, self.y_total_train, sample_weight=self.weights_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        
        # Predictions
        y_pred_train = best_model.predict(self.X_train)
        y_pred_test = best_model.predict(self.X_test)
        
        # Metrics
        train_mae = mean_absolute_error(self.y_total_train, y_pred_train)
        test_mae = mean_absolute_error(self.y_total_test, y_pred_test)
        test_r2 = r2_score(self.y_total_test, y_pred_test)
        
        print(f"✅ Best parameters: {grid_search.best_params_}")
        print(f"📊 Training MAE: {train_mae:.2f}")
        print(f"📊 Test MAE: {test_mae:.2f}")
        print(f"📊 Test R²: {test_r2:.3f}")
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': self.features,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔝 Top 5 Features:")
        for _, row in importance_df.head().iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
        
        # Store results
        self.models['total_points'] = best_model
        self.results['total_points'] = {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'feature_importance': importance_df
        }
        
        return best_model
    
    def tune_win_margin_model(self):
        """Retune the win margin prediction model"""
        if self.y_margin is None:
            print("⚠️ No margin of victory target variable found")
            return
            
        print("\n🎯 Tuning Win Margin Model...")
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [15, 20, 25],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4]
        }
        
        # Grid search
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1
        )
        
        print("⚙️ Running hyperparameter optimization...")
        grid_search.fit(self.X_train, self.y_margin_train, sample_weight=self.weights_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        
        # Predictions with confidence intervals
        all_tree_preds = np.array([tree.predict(self.X_test) for tree in best_model.estimators_])
        y_pred_test = np.mean(all_tree_preds, axis=0)
        pred_std = np.std(all_tree_preds, axis=0)
        
        # Metrics
        y_pred_train = best_model.predict(self.X_train)
        train_mae = mean_absolute_error(self.y_margin_train, y_pred_train)
        test_mae = mean_absolute_error(self.y_margin_test, y_pred_test)
        test_r2 = r2_score(self.y_margin_test, y_pred_test)
        
        print(f"✅ Best parameters: {grid_search.best_params_}")
        print(f"📊 Training MAE: {train_mae:.2f}")
        print(f"📊 Test MAE: {test_mae:.2f}")
        print(f"📊 Test R²: {test_r2:.3f}")
        print(f"📊 Average prediction std: {np.mean(pred_std):.2f}")
        
        # Store results
        self.models['win_margin'] = best_model
        self.results['win_margin'] = {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'prediction_std': pred_std
        }
        
        return best_model
    
    def save_models(self):
        """Save the tuned models"""
        print("\n💾 Saving improved models...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, model in self.models.items():
            filename = f"src/modeling/retuned_{model_name}_model_{timestamp}.pkl"
            joblib.dump(model, filename)
            print(f"✅ Saved {model_name} model to {filename}")
    
    def generate_report(self):
        """Generate improvement report"""
        print("\n📋 MODEL RETUNING REPORT")
        print("=" * 50)
        
        for model_name, results in self.results.items():
            print(f"\n{model_name.upper()} MODEL:")
            print(f"  Test MAE: {results['test_mae']:.2f}")
            print(f"  Test R²:  {results['test_r2']:.3f}")
            
            if 'feature_importance' in results:
                print(f"  Top feature: {results['feature_importance'].iloc[0]['feature']}")
        
        print(f"\n🕒 Retuning completed at: {datetime.now()}")
        print("✅ Models ready for deployment!")

def main():
    """Main execution function"""
    print("🏈 NCAA Football Model Re-tuning")
    print("=" * 40)
    
    retuner = ModelRetuner()
    
    # Phase 1: Data loading and preparation
    if not retuner.load_data():
        return
    
    retuner.prepare_features()
    retuner.create_train_test_split()
    
    # Phase 2: Model tuning
    retuner.tune_total_points_model()
    retuner.tune_win_margin_model()
    
    # Phase 3: Save and report
    retuner.save_models()
    retuner.generate_report()
    
    print("\n🎉 Re-tuning complete! Your models are now improved.")

if __name__ == "__main__":
    main()
