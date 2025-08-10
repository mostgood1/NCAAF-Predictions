"""
Quick Enhanced Prediction Generation
Uses our newly enhanced models to generate better predictions
"""

import sys
import os
sys.path.append('src')

# Import our enhanced models
from modeling.model_total_points import predict_total_points
from modeling.model_team_points import predict_team_points
from modeling.model_win_margin import predict_win_margin

import pandas as pd
import numpy as np

def generate_enhanced_predictions():
    print("🚀 Generating Enhanced 2025 Predictions")
    print("=" * 45)
    
    # Load the 2025 schedule
    print("📅 Loading 2025 schedule...")
    try:
        schedule = pd.read_csv("src/data/college_football_schedule_2025.csv")
        print(f"✅ Loaded {len(schedule)} games")
    except Exception as e:
        print(f"❌ Error loading schedule: {e}")
        return None
    
    enhanced_predictions = []
    
    print("🔮 Generating enhanced predictions...")
    for index, game in schedule.iterrows():
        try:
            # Use our enhanced models
            total_points = predict_total_points(game['home_team'], game['away_team'])
            home_points = predict_team_points(game['home_team'], game['away_team'], is_home=True)
            away_points = predict_team_points(game['away_team'], game['home_team'], is_home=False)
            win_margin = predict_win_margin(game['home_team'], game['away_team'])
            
            prediction = {
                'season': game['season'],
                'week': game['week'],
                'start_date': game['start_date'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'venue': game['venue'],
                'conference_game': game['conference_game'],
                'predicted_total_points': total_points,
                'predicted_home_points': home_points,
                'predicted_away_points': away_points,
                'predicted_win_margin': win_margin
            }
            enhanced_predictions.append(prediction)
            
            if (index + 1) % 500 == 0:
                print(f"  ✅ Processed {index + 1} games...")
                
        except Exception as e:
            print(f"⚠️  Error predicting game {index}: {e}")
            continue
    
    # Convert to DataFrame
    enhanced_df = pd.DataFrame(enhanced_predictions)
    
    # Save the enhanced predictions
    output_file = "src/data/college_football_schedule_2025_predicted_totals.csv"
    enhanced_df.to_csv(output_file, index=False)
    print(f"✅ Saved enhanced predictions to {output_file}")
    
    # Show sample of improved predictions
    print("\n🎯 Sample Enhanced Predictions:")
    print("Week | Home Team      vs Away Team      | Total | Home | Away | Margin")
    print("-" * 75)
    
    for i in range(min(10, len(enhanced_df))):
        row = enhanced_df.iloc[i]
        home = str(row['home_team'])[:12]
        away = str(row['away_team'])[:12]
        print(f"{row['week']:2d}   | {home:12s} vs {away:12s} | {row['predicted_total_points']:5.1f} | {row['predicted_home_points']:4.1f} | {row['predicted_away_points']:4.1f} | {row['predicted_win_margin']:+5.1f}")
    
    print(f"\n🎉 Enhanced predictions complete!")
    print(f"Your models now use 4x more trees with advanced techniques!")
    
    return enhanced_df

if __name__ == "__main__":
    enhanced_predictions = generate_enhanced_predictions()
