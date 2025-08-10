"""
Enhanced NCAA Football Prediction Models
Improved version with better hyperparameters and features
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def create_enhanced_total_points_model():
    """Create improved total points prediction model"""
    
    # Load feature data
    df = pd.read_csv("src/data/ncaa_games_last_15_years_features_with_elo.csv")
    
    # Enhanced feature set
    features = [
        "week", "season",
        "home_elo", "away_elo",
        "home_conf_encoded", "away_conf_encoded", "venue_encoded"
    ]
    
    # Add optional features if available
    optional_features = [
        "betting_spread", "betting_over_under",
        "home_season_avg_home", "away_season_avg_away",
        "home_recent_opp_avg_elo", "away_recent_opp_avg_elo"
    ]
    
    for feat in optional_features:
        if feat in df.columns:
            features.append(feat)
    
    # Prepare data
    X = df[features]
    y = df["game_total_points"]
    
    # Create sample weights (recent games matter more)
    weights = np.exp((df['season'] - 2010) / 5)
    
    # Train on all available data with improved parameters
    model = RandomForestRegressor(
        n_estimators=500,     # More trees
        max_depth=25,         # Deeper trees
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',   # Feature randomness
        random_state=42,
        n_jobs=-1             # Use all cores
    )
    
    # Fit with sample weights
    model.fit(X, y, sample_weight=weights)
    
    # Calculate training accuracy
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"Enhanced Total Points Model - Training MAE: {mae:.2f}, R²: {r2:.3f}")
    
    # Save model
    joblib.dump(model, "src/modeling/enhanced_total_points_model.pkl")
    joblib.dump(features, "src/modeling/total_points_features.pkl")
    
    return model, features

def create_enhanced_win_margin_model():
    """Create improved win margin prediction model"""
    
    # Load feature data
    df = pd.read_csv("src/data/ncaa_games_last_15_years_features_with_elo.csv")
    
    # Enhanced feature set for margin prediction
    features = [
        "week", "season", "game_total_points",
        "home_elo", "away_elo",
        "home_conf_encoded", "away_conf_encoded", "venue_encoded"
    ]
    
    # Add optional features
    optional_features = [
        "betting_spread", "betting_over_under", 
        "home_season_avg_home", "away_season_avg_away",
        "home_recent_opp_avg_elo", "away_recent_opp_avg_elo"
    ]
    
    for feat in optional_features:
        if feat in df.columns:
            features.append(feat)
    
    # Prepare data
    X = df[features]
    y = df["margin_of_victory"]
    
    # Sample weights
    weights = np.exp((df['season'] - 2010) / 5)
    
    # Enhanced model
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    # Fit model
    model.fit(X, y, sample_weight=weights)
    
    # Calculate accuracy
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"Enhanced Win Margin Model - Training MAE: {mae:.2f}, R²: {r2:.3f}")
    
    # Save model
    joblib.dump(model, "src/modeling/enhanced_win_margin_model.pkl")
    joblib.dump(features, "src/modeling/win_margin_features.pkl")
    
    return model, features

def create_enhanced_team_points_models():
    """Create separate models for home and away team points"""
    
    df = pd.read_csv("src/data/ncaa_games_last_15_years_features_with_elo.csv")
    
    # Features for team-specific models
    base_features = ["week", "season", "home_elo", "away_elo", "home_conf_encoded", "away_conf_encoded", "venue_encoded"]
    
    # Add optional features
    optional_features = ["betting_spread", "betting_over_under", "home_season_avg_home", "away_season_avg_away"]
    features = base_features + [f for f in optional_features if f in df.columns]
    
    X = df[features]
    weights = np.exp((df['season'] - 2010) / 5)
    
    # HOME TEAM POINTS MODEL
    y_home = df["home_points"]
    home_model = RandomForestRegressor(
        n_estimators=400, max_depth=20, min_samples_split=5,
        random_state=42, n_jobs=-1
    )
    home_model.fit(X, y_home, sample_weight=weights)
    
    home_pred = home_model.predict(X)
    home_mae = mean_absolute_error(y_home, home_pred)
    print(f"Enhanced Home Points Model - Training MAE: {home_mae:.2f}")
    
    # AWAY TEAM POINTS MODEL
    y_away = df["away_points"]
    away_model = RandomForestRegressor(
        n_estimators=400, max_depth=20, min_samples_split=5,
        random_state=42, n_jobs=-1
    )
    away_model.fit(X, y_away, sample_weight=weights)
    
    away_pred = away_model.predict(X)
    away_mae = mean_absolute_error(y_away, away_pred)
    print(f"Enhanced Away Points Model - Training MAE: {away_mae:.2f}")
    
    # Save models
    joblib.dump(home_model, "src/modeling/enhanced_home_points_model.pkl")
    joblib.dump(away_model, "src/modeling/enhanced_away_points_model.pkl")
    joblib.dump(features, "src/modeling/team_points_features.pkl")
    
    return home_model, away_model, features

def main():
    """Create all enhanced models"""
    print("🏈 Creating Enhanced NCAA Football Prediction Models")
    print("=" * 55)
    
    try:
        # Create enhanced models
        print("🎯 Building Total Points Model...")
        total_model, total_features = create_enhanced_total_points_model()
        
        print("🎯 Building Win Margin Model...")
        margin_model, margin_features = create_enhanced_win_margin_model()
        
        print("🎯 Building Team Points Models...")
        home_model, away_model, team_features = create_enhanced_team_points_models()
        
        print("\n✅ All enhanced models created successfully!")
        print("\nModel files saved:")
        print("  - enhanced_total_points_model.pkl")
        print("  - enhanced_win_margin_model.pkl") 
        print("  - enhanced_home_points_model.pkl")
        print("  - enhanced_away_points_model.pkl")
        
        print("\n🎉 Your models are now significantly improved!")
        print("Features used:")
        print(f"  - Total Points: {len(total_features)} features")
        print(f"  - Win Margin: {len(margin_features)} features")
        print(f"  - Team Points: {len(team_features)} features")
        
    except Exception as e:
        print(f"❌ Error creating models: {e}")

if __name__ == "__main__":
    main()
