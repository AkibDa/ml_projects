import pandas as pd
import numpy as np
import os

# Load data
train_dir = "data/train/"

dfs = []
for file in os.listdir(train_dir):
  if file.endswith(".csv"):
    path = os.path.join(train_dir, file)
    df = pd.read_csv(path)
    dfs.append(df)

train = pd.concat(dfs, ignore_index=True)

fps = 10  

start_df = train[train["frame_id"] == 1].copy()

predictions = []

for _, row in start_df.iterrows():
  x0, y0 = row["x"], row["y"]
  s = row["s"]
  dir_rad = np.deg2rad(row["dir"])
  
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
