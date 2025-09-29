Curated data snapshot for 2025 UI (used by app in production)

Included files
- college_football_schedule_2025_predicted_totals_enhanced_20250929T134520Z.csv
  Latest enhanced predictions (through Week 6+). Selected by timestamp in filename.
- college_football_schedule_2025_predicted_totals_enhanced_with_scores.csv
  Enhanced predictions joined with final scores for completed games. Drives accuracy/YTD.
- college_football_betting_lines_2025.csv
  Consensus/market lines for 2025 season (spreads, totals) used to compute ATS/OU edges.
- team_conferences.csv
  Team to conference mapping used to determine FBS vs FCS and to scope metrics to FBS-involved games.

Notes
- Additional large or intermediate CSVs remain untracked to keep the repo lean.
- App code prefers the latest enhanced predictions file when multiple exist. Keeping the latest timestamped file here ensures Week 6+ data is visible after deploy.
- If regenerating data, please update this curated file set and commit with a clear message, e.g., "data: refresh 2025 snapshot (YYYY-MM-DD)".
