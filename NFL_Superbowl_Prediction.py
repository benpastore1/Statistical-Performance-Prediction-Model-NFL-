
import pandas as pd
import sys
import os

# Check if CSV files exist
required_files = ["Games.csv", "Summary.csv", "Teams.csv"]
for file in required_files:
    if not os.path.exists(file):
        print(f"ERROR: {file} not found in current directory!")
        print(f"Current directory: {os.getcwd()}")
        sys.exit(1)

try:
    games = pd.read_csv("Games.csv")
    summary = pd.read_csv("Summary.csv")
    teams = pd.read_csv("Teams.csv")
except FileNotFoundError as e:
    print(f"ERROR: Could not find required CSV file: {e}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR reading CSV files: {e}")
    sys.exit(1)

games = games.rename(columns={
    "Points Scored": "points_scored",
    "Opponent Defensive Tier": "opp_def_tier",
    "Game #": "game_num"
})

summary = summary.rename(columns={
    "Tier 1 Avg Points": "tier1_avg",
    "Tier 2 Avg Points": "tier2_avg",
    "Tier 3 Avg Points": "tier3_avg",
    "Tier 4 Avg Points": "tier4_avg"
})
teams_opp = teams[["Team", "Average Points Allowed Per Game"]].rename(columns={
    "Average Points Allowed Per Game": "opp_avg_points_allowed"
})


games = games.merge(
    summary,
    on="Team",
    how="left"
)
games = games.merge(
    teams_opp,
    left_on="Opponent",
    right_on="Team",
    how="left",
    suffixes=("", "_opp")
).drop(columns=["Team_opp"])

# Convert "Weighted Playoff Offense Score" to numeric, replacing '--' with NaN
games["Weighted Playoff Offense Score"] = pd.to_numeric(
    games["Weighted Playoff Offense Score"], 
    errors='coerce'
)

# Drop rows with missing values in features
games = games.dropna(subset=["Weighted Playoff Offense Score", "Offensive Momentum", "opp_avg_points_allowed", "opp_def_tier"])

print(f"Loaded {len(games)} games for training")
print(f"Features: {', '.join(['Weighted Playoff Offense Score', 'Offensive Momentum', 'Opponent Avg Points Allowed', 'Opponent Defensive Tier'])}")

# ====================
#  model training
# ====================

# Predict Patriots score vs Seahawks
try:
    from sklearn.linear_model import LinearRegression  # pyright: ignore[reportMissingImports]
except ImportError as e:
    print(f"ERROR: scikit-learn is not installed. Please install it with: pip install scikit-learn")
    print(f"Error details: {e}")
    sys.exit(1)

# ----- MODEL TRAINING -----

features = [
    "Weighted Playoff Offense Score",
    "Offensive Momentum",
    "opp_avg_points_allowed",
    "opp_def_tier"
]

try:
    X = games[features]
    y = games["points_scored"]
    
    # Check for missing features
    missing_features = [f for f in features if f not in games.columns]
    if missing_features:
        print(f"ERROR: Missing features in data: {missing_features}")
        print(f"Available columns: {list(games.columns)}")
        sys.exit(1)
    
    model = LinearRegression()
    model.fit(X, y)
    
    print("MODEL TRAINED")
except KeyError as e:
    print(f"ERROR: Missing column in data: {e}")
    print(f"Available columns: {list(games.columns)}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR during model training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Calculate mean Weighted Playoff Offense Score from training data as fallback
mean_weighted_score = X["Weighted Playoff Offense Score"].mean()

# Get Patriots offensive stats and Seahawks defensive stats
try:
    pat_off = summary[summary["Team"] == "New England Patriots"].iloc[0]
    sea_def = teams[teams["Team"] == "Seattle Seahawks"].iloc[0]
    
    # Get Seahawks offensive stats and Patriots defensive stats
    sea_off = summary[summary["Team"] == "Seattle Seahawks"].iloc[0]
    pat_def = teams[teams["Team"] == "New England Patriots"].iloc[0]
except IndexError as e:
    print(f"ERROR: Could not find team data. Please check that 'New England Patriots' and 'Seattle Seahawks' exist in the data.")
    print(f"Available teams in summary: {summary['Team'].unique()[:10]}")
    print(f"Available teams in teams: {teams['Team'].unique()[:10]}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR retrieving team data: {e}")
    sys.exit(1)

# Helper function to get Weighted Playoff Offense Score with fallback
def get_weighted_score(team_off, team_def_data):
    score = pd.to_numeric(team_off["Weighted Playoff Offense Score"], errors='coerce')
    if pd.isna(score):
        # Use team's average points scored as fallback
        score = team_def_data["Average Points Scored Per Game"]
    return score

# Patriots prediction dataframe - ensure column order matches features
pat_weighted_score = get_weighted_score(pat_off, teams[teams["Team"] == "New England Patriots"].iloc[0])
pat_df = pd.DataFrame([{
    "Weighted Playoff Offense Score": pat_weighted_score,
    "Offensive Momentum": pat_off["Offensive Momentum"],
    "opp_avg_points_allowed": sea_def["Average Points Allowed Per Game"],
    "opp_def_tier": sea_def["Defensive Tier Rank"]
}], columns=features)

# Predict Patriots score
pat_pred = model.predict(pat_df)[0]

print(f"\n{'='*60}")
print(f"PREDICTION: New England Patriots vs Seattle Seahawks")
print(f"{'='*60}")

# ---- Same for Seahawks ----

sea_weighted_score = get_weighted_score(sea_off, teams[teams["Team"] == "Seattle Seahawks"].iloc[0])
sea_df = pd.DataFrame([{
    "Weighted Playoff Offense Score": sea_weighted_score,
    "Offensive Momentum": sea_off["Offensive Momentum"],
    "opp_avg_points_allowed": pat_def["Average Points Allowed Per Game"],
    "opp_def_tier": pat_def["Defensive Tier Rank"]
}], columns=features)

sea_pred = model.predict(sea_df)[0]

# Display predictions
print(f"\nNew England Patriots: {pat_pred:.1f} points")
print(f"Seattle Seahawks:     {sea_pred:.1f} points")

# Determine winner
if pat_pred > sea_pred:
    winner = "New England Patriots"
    margin = pat_pred - sea_pred
elif sea_pred > pat_pred:
    winner = "Seattle Seahawks"
    margin = sea_pred - pat_pred
else:
    winner = "Tie"
    margin = 0

print(f"\n{'='*60}")
if winner != "Tie":
    print(f"PREDICTED WINNER: {winner}")
    print(f"Predicted Margin: {margin:.1f} points")
else:
    print(f"PREDICTED RESULT: {winner}")
print(f"{'='*60}\n")