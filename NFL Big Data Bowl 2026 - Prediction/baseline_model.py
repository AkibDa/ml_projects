import pandas as pd
import numpy as np
import os

train_dir = "data/train/"

dfs = []
for file in os.listdir(train_dir):
  if file.endswith(".csv"):
    path = os.path.join(train_dir, file)
    df = pd.read_csv(path)
    dfs.append(df)

train = pd.concat(dfs, ignore_index=True)

fps = 10  

median_frames = int(train["num_frames_output"].median())
train["num_frames_output"] = train["num_frames_output"].fillna(median_frames)

start_df = train[train["frame_id"] == 1].copy()

start_df = start_df.dropna(subset=["s", "dir"])

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
    x_t = x0 + vx * (t / fps)
    y_t = y0 + vy * (t / fps)
    
    pred_id = f"{row['game_id']}_{row['play_id']}_{row['nfl_id']}_{frame_id}"
    predictions.append([pred_id, x_t, y_t])

sub = pd.DataFrame(predictions, columns=["id", "x", "y"])

sub.to_csv("baseline_constant_velocity.csv", index=False)
print("Baseline file saved:", sub.shape)
