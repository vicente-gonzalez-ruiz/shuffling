import numpy as np

def random(vol, mean=0.0, std_dev=1.0):
  depth, height, width = vol.shape[:3]
  z_coords, y_coords, x_coords = np.meshgrid(range(depth), range(height), range(width), indexing="ij")
  flattened_x_coords = x_coords.flatten()
  flattened_y_coords = y_coords.flatten()
  flattened_z_coords = z_coords.flatten()
  #print(np.max(flattened_z_coords), np.max(flattened_y_coords), np.max(flattened_x_coords))
  #print(flattened_x_coords.dtype)
  displacements_x = np.random.normal(mean, std_dev, flattened_x_coords.shape).astype(np.int32)
  displacements_y = np.random.normal(mean, std_dev, flattened_y_coords.shape).astype(np.int32)
  displacements_z = np.random.normal(mean, std_dev, flattened_z_coords.shape).astype(np.int32)
  #_d = 5
  #displacements_x = np.random.uniform(low=-_d, high=_d, size=flattened_x_coords.shape).astype(np.int32)
  #displacements_y = np.random.uniform(low=-_d, high=_d, size=flattened_y_coords.shape).astype(np.int32)
  #displacements_z = np.random.uniform(low=-_d, high=_d, size=flattened_z_coords.shape).astype(np.int32)
  print("min displacements", np.min(displacements_z), np.min(displacements_y), np.min(displacements_x))
  print("average abs(displacements)", np.average(np.abs(displacements_z)), np.average(np.abs(displacements_y)), np.average(np.abs(displacements_x)))
  print("max displacements", np.max(displacements_z), np.max(displacements_y), np.max(displacements_x))
  randomized_x_coords = flattened_x_coords + displacements_x
  randomized_y_coords = flattened_y_coords + displacements_y
  randomized_z_coords = flattened_z_coords + displacements_z
  #print("max displacements", np.max(randomized_z_coords), np.max(randomized_y_coords), np.max(randomized_x_coords))
  #randomized_x_coords = np.mod(randomized_x_coords, width)
  #randomized_y_coords = np.mod(randomized_y_coords, height)
  #randomized_z_coords = np.mod(randomized_z_coords, depth)
  randomized_x_coords = np.clip(randomized_x_coords, 0, width - 1) # Clip the randomized coordinates to stay within image bounds
  randomized_y_coords = np.clip(randomized_y_coords, 0, height - 1)
  randomized_z_coords = np.clip(randomized_z_coords, 0, depth - 1)
  #print(np.max(randomized_z_coords), np.max(randomized_y_coords), np.max(randomized_x_coords))
  #randomized_vol = np.ones_like(vol)*np.average(vol) #np.zeros_like(vol)
  randomized_vol = np.zeros_like(vol)
  #randomized_vol[...] = vol
  #randomized_vol[...] = 128
  #print("z", randomized_z_coords)
  #print("y", randomized_y_coords)
  #print("x", randomized_x_coords)
  #print("z", flattened_z_coords)
  #print("y", flattened_y_coords)
  #print("x", flattened_x_coords)
  randomized_vol[randomized_z_coords, randomized_y_coords, randomized_x_coords] = vol[flattened_z_coords, flattened_y_coords, flattened_x_coords]
  return randomized_vol
