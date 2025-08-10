"""
Quick Model Evaluation and Improvement
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def evaluate_current_models():
    print("🏈 NCAA Football Model Evaluation")
    print("=" * 40)
    
    try:
        # Load data
        print("📊 Loading data...")
        df = pd.read_csv("src/data/ncaa_games_last_15_years_features_with_elo.csv")
        print(f"✅ Loaded {len(df)} games")
        
        # Check available features
        available_features = []
        potential_features = [
            "week", "season", "home_elo", "away_elo",
            "home_conf_encoded", "away_conf_encoded", "venue_encoded",
            "betting_spread", "betting_over_under",
            "home_season_avg_home", "away_season_avg_away"
        ]
        
        for feat in potential_features:
            if feat in df.columns:
                available_features.append(feat)
        
        print(f"✅ Using {len(available_features)} features")
        
        # Prepare data
        X = df[available_features]
        y_total = df["game_total_points"]
        y_margin = df["margin_of_victory"]
        
        # Train/test split (temporal)
        train_mask = df['season'] < 2024
        test_mask = df['season'] == 2024
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_total_train, y_total_test = y_total[train_mask], y_total[test_mask]
        y_margin_train, y_margin_test = y_margin[train_mask], y_margin[test_mask]
        
        print(f"Training games: {len(X_train)}")
        print(f"Test games: {len(X_test)}")
        
        # TOTAL POINTS MODEL
        print("\n🎯 Total Points Model:")
        rf_total = RandomForestRegressor(
            n_estimators=300,  # Increased from 100
            max_depth=20,      # Added depth control
            min_samples_split=5,
            random_state=42
        )
        rf_total.fit(X_train, y_total_train)
        
        y_pred_total = rf_total.predict(X_test)
        mae_total = mean_absolute_error(y_total_test, y_pred_total)
        r2_total = r2_score(y_total_test, y_pred_total)
        
        print(f"  MAE: {mae_total:.2f} points")
        print(f"  R²:  {r2_total:.3f}")
        
        # WIN MARGIN MODEL
        print("\n🎯 Win Margin Model:")
        rf_margin = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            random_state=42
        )
        rf_margin.fit(X_train, y_margin_train)
        
        y_pred_margin = rf_margin.predict(X_test)
        mae_margin = mean_absolute_error(y_margin_test, y_pred_margin)
        r2_margin = r2_score(y_margin_test, y_pred_margin)
        
        print(f"  MAE: {mae_margin:.2f} points")
        print(f"  R²:  {r2_margin:.3f}")
        
        # Feature importance
        print("\n🔝 Most Important Features (Total Points):")
        importances = rf_total.feature_importances_
        feature_imp = sorted(zip(available_features, importances), key=lambda x: x[1], reverse=True)
        for feat, imp in feature_imp[:5]:
            print(f"  {feat}: {imp:.3f}")
        
        # Sample predictions
        print("\n📋 Sample Predictions (First 5 test games):")
        for i in range(min(5, len(y_pred_total))):
            actual_total = y_total_test.iloc[i]
            pred_total = y_pred_total[i]
            actual_margin = y_margin_test.iloc[i]
            pred_margin = y_pred_margin[i]
            print(f"  Game {i+1}: Total {actual_total:.0f} → {pred_total:.0f}, Margin {actual_margin:.0f} → {pred_margin:.0f}")
        
        print("\n✅ Model evaluation complete!")
        print("📈 These are improved baseline models with:")
        print("   - Increased trees (300 vs 100)")
        print("   - Better depth control")
        print("   - Temporal validation")
        
        return rf_total, rf_margin, mae_total, mae_margin
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None, None, None

if __name__ == "__main__":
    evaluate_current_models()
