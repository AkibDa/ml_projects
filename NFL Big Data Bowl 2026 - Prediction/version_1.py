import pandas as pd
import numpy as np
import os

BASE_DIR = "/kaggle/input/nfl-big-data-bowl-2026-prediction/"
TEST_INPUT_FILE = os.path.join(BASE_DIR, "test_input.csv")

try:
  test_combined = pd.read_csv(TEST_INPUT_FILE)
  print("Successfully loaded test_input.csv")
  print(f"Data shape: {test_combined.shape}")
except FileNotFoundError:
  print(f"Error: Could not find the file at {TEST_INPUT_FILE}.")
  print("Please ensure the file name is correct and it exists in the input directory.")
  test_combined = pd.DataFrame()


if not test_combined.empty:
  fps = 10
  
  median_frames = 25
  test_combined["num_frames_output"] = test_combined["num_frames_output"].fillna(median_frames)

  start_df = test_combined[test_combined["frame_id"] == 1].copy()

  start_df = start_df.dropna(subset=["s", "dir"])
  print(f"Found {len(start_df)} player starting positions to predict from.")

  predictions = []

  for _, row in start_df.iterrows():
    x0, y0 = row["x"], row["y"]
    s = row["s"]
    dir_deg = row["dir"]

    if row["play_direction"] == "left":
      dir_deg = (dir_deg + 180) % 360

    dir_rad = np.deg2rad(dir_deg)

    vx = s * np.cos(dir_rad)
    vy = s * np.sin(dir_rad)

    num_frames = int(row["num_frames_output"])

    for t in range(1, num_frames + 1):
      frame_id = row["frame_id"] + t
      
      time_elapsed = t / fps
      x_t = x0 + vx * time_elapsed
      y_t = y0 + vy * time_elapsed

      pred_id = f"{int(row['game_id'])}_{int(row['play_id'])}_{int(row['nfl_id'])}_{frame_id}"
      predictions.append([pred_id, x_t, y_t])

  if predictions:
    submission_df = pd.DataFrame(predictions, columns=["id", "x", "y"])

    submission_df.to_csv("submission.csv", index=False)

    print("\nSubmission file successfully created!")
    print(f"File shape: {submission_df.shape}")
    print("First 5 rows of submission file:")
    print(submission_df.head())
  else:
    print("No predictions were generated. The resulting dataframe is empty.")
    pd.DataFrame(columns=["id", "x", "y"]).to_csv("submission.csv", index=False)

else:
  print("Test data could not be loaded or is empty. Cannot generate submission file.")
  pd.DataFrame(columns=["id", "x", "y"]).to_csv("submission.csv", index=False)

