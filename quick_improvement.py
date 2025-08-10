"""
Quick Model Improvement Script
Updates your existing models with better hyperparameters
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def quick_model_improvement():
    print("🚀 NCAA Football Quick Model Improvement")
    print("=" * 42)
    
    try:
        # Load your existing data
        df = pd.read_csv("src/data/ncaa_games_last_15_years_features_with_elo.csv")
        print(f"✅ Loaded {len(df)} historical games")
        
        # Use available features
        features = ["week", "season", "home_elo", "away_elo", "home_conf_encoded", "away_conf_encoded", "venue_encoded"]
        
        # Add optional features if they exist
        optional = ["betting_spread", "betting_over_under", "home_season_avg_home", "away_season_avg_away"]
        for feat in optional:
            if feat in df.columns:
                features.append(feat)
        
        print(f"✅ Using {len(features)} features")
        
        X = df[features]
        
        # Add recency weights (recent games matter more)
        weights = np.exp((df['season'] - 2010) / 4)
        
        # Test on 2024 data
        train_mask = df['season'] < 2024
        test_mask = df['season'] == 2024
        
        X_train, X_test = X[train_mask], X[test_mask]
        weights_train = weights[train_mask]
        
        print(f"Training on {len(X_train)} games, testing on {len(X_test)} games")
        
        # IMPROVED TOTAL POINTS MODEL
        print("\n🎯 Improving Total Points Model...")
        
        # Old model (baseline)
        old_model = RandomForestRegressor(n_estimators=100, random_state=42)
        old_model.fit(X_train, df["game_total_points"][train_mask])
        old_pred = old_model.predict(X_test)
        old_mae = mean_absolute_error(df["game_total_points"][test_mask], old_pred)
        
        # NEW IMPROVED MODEL
        new_model = RandomForestRegressor(
            n_estimators=400,      # 4x more trees
            max_depth=22,          # Better depth
            min_samples_split=6,   # Better splits
            min_samples_leaf=3,    # Better leaves
            max_features='sqrt',   # Feature randomness
            random_state=42,
            n_jobs=-1              # Use all CPU cores
        )
        new_model.fit(X_train, df["game_total_points"][train_mask], sample_weight=weights_train)
        new_pred = new_model.predict(X_test)
        new_mae = mean_absolute_error(df["game_total_points"][test_mask], new_pred)
        new_r2 = r2_score(df["game_total_points"][test_mask], new_pred)
        
        improvement = ((old_mae - new_mae) / old_mae) * 100
        
        print(f"  📊 Old Model MAE: {old_mae:.2f} points")
        print(f"  📊 NEW Model MAE: {new_mae:.2f} points")
        print(f"  📊 NEW Model R²:  {new_r2:.3f}")
        print(f"  🎉 Improvement: {improvement:.1f}% better!")
        
        # IMPROVED WIN MARGIN MODEL
        print("\n🎯 Improving Win Margin Model...")
        
        margin_features = features + ["game_total_points"] if "game_total_points" in df.columns else features
        X_margin = df[margin_features]
        X_margin_train, X_margin_test = X_margin[train_mask], X_margin[test_mask]
        
        # Old margin model
        old_margin_model = RandomForestRegressor(n_estimators=100, random_state=42)
        old_margin_model.fit(X_margin_train, df["margin_of_victory"][train_mask])
        old_margin_pred = old_margin_model.predict(X_margin_test)
        old_margin_mae = mean_absolute_error(df["margin_of_victory"][test_mask], old_margin_pred)
        
        # NEW IMPROVED MARGIN MODEL
        new_margin_model = RandomForestRegressor(
            n_estimators=400,
            max_depth=20,
            min_samples_split=6,
            min_samples_leaf=3,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        new_margin_model.fit(X_margin_train, df["margin_of_victory"][train_mask], sample_weight=weights_train)
        new_margin_pred = new_margin_model.predict(X_margin_test)
        new_margin_mae = mean_absolute_error(df["margin_of_victory"][test_mask], new_margin_pred)
        new_margin_r2 = r2_score(df["margin_of_victory"][test_mask], new_margin_pred)
        
        margin_improvement = ((old_margin_mae - new_margin_mae) / old_margin_mae) * 100
        
        print(f"  📊 Old Model MAE: {old_margin_mae:.2f} points")
        print(f"  📊 NEW Model MAE: {new_margin_mae:.2f} points") 
        print(f"  📊 NEW Model R²:  {new_margin_r2:.3f}")
        print(f"  🎉 Improvement: {margin_improvement:.1f}% better!")
        
        # Feature importance
        print(f"\n🔝 Most Important Features:")
        importances = new_model.feature_importances_
        feature_ranking = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
        for i, (feat, imp) in enumerate(feature_ranking[:5]):
            print(f"  {i+1}. {feat}: {imp:.3f}")
        
        print(f"\n✅ SUMMARY:")
        print(f"   🎯 Total Points Model: {improvement:.1f}% improvement")
        print(f"   🎯 Win Margin Model: {margin_improvement:.1f}% improvement")
        print(f"   📈 Using recency weighting and 400 trees")
        print(f"   🚀 Ready to generate better 2025 predictions!")
        
        return new_model, new_margin_model, features, margin_features
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

if __name__ == "__main__":
    total_model, margin_model, total_features, margin_features = quick_model_improvement()
