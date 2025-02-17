# Sin terminar

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

# https://stackoverflow.com/questions/62436299/how-to-lightly-shuffle-a-list-in-python
orderliness = 0.75

def tuplify(x, y):
  return (orderliness * y + np.random.normal(0, 1), x)

############

def shake(x, y, std_dev=1.0):
  displacements = np.random.normal(0, std_dev, len(x))
  #print(f"{np.min(displacements):.2f} {np.average(np.abs(displacements)):.2f} {np.max(displacements):.2f}", end=' ')
  return np.stack((y + displacements, x), axis=1)

############

def randomize(vol, mean=0.0, std_dev=1.0):
  print(vol.shape)
  print(std_dev)
  randomized_vol = np.empty_like(vol)
  
  # Randomization in X
  #values = np.arange(1, vol.shape[2]+1).astype(np.int32)
  values = np.arange(vol.shape[2]).astype(np.int32)
  for z in range(vol.shape[0]):
    print(z, end=' ', flush=True)
    for y in range(vol.shape[1]):
      #pairs = np.array(list(map(tuplify, values, range(len(values)))), dtype=np.int32)
      pairs = shake(values, np.arange(len(values)), std_dev).astype(np.int32)
      pairs = pairs[pairs[:, 0].argsort()]
      randomized_vol[z, y, values] = vol[z, y, pairs[:, 1]]

  # Randomization in Y
  values = np.arange(vol.shape[1]).astype(np.int32)
  for z in range(vol.shape[0]):
    print(z, end=' ', flush=True)
    for x in range(vol.shape[2]):
      #pairs = np.array(list(map(tuplify, values, range(len(values)))), dtype=np.int32)
      pairs = shake(values, np.arange(len(values)), std_dev).astype(np.int32)
      pairs = pairs[pairs[:, 0].argsort()]
      randomized_vol[z, values, x] = vol[z, pairs[:, 1], x]

  # Randomization in Z
  values = np.arange(vol.shape[0]).astype(np.int32)
  for y in range(vol.shape[1]):
    print(y, end=' ', flush=True)
    for x in range(vol.shape[2]):
      #pairs = np.array(list(map(tuplify, values, range(len(values)))), dtype=np.int32)
      pairs = shake(values, np.arange(len(values)), std_dev).astype(np.int32)
      pairs = pairs[pairs[:, 0].argsort()]
      randomized_vol[values, y, x] = vol[pairs[:, 1], y , x]

  return randomized_vol

def project_A_to_B(A, B, estimator, block_size):
  output_vz, output_vy, output_vx, output_confidence = estimator.calculate_flow(
    A, B,
    start_point=(0, 0, 0),
    total_vol=(A.shape[0], A.shape[1], A.shape[2]),
    sub_volume=block_size,
    overlap=(8, 8, 8),
    threadsperblock=(8, 8, 8)
  )
  print("min flow", np.min(output_vx), np.min(output_vy), np.min(output_vz))
  print("average abs(flow)", np.average(np.abs(output_vx)), np.average(np.abs(output_vy)), np.average(np.abs(output_vz)))
  print("max flow", np.max(output_vx), np.max(output_vy), np.max(output_vz))
  projection = opticalflow3D.helpers.generate_inverse_image(A, output_vx, output_vy, output_vz, use_gpu=False)
  return projection

#randomize(np.zeros(shape=(3,3,3)))
#quit()
vol_MRC = mrcfile.open(args.input)
noisy = vol_MRC.data
block_size = (noisy.shape[0]//2, noisy.shape[1]//2, noisy.shape[2]//2)

for i in range(8):
  print(args[i+1])
  with mrcfile.new(args[i+1], overwrite=True) as mrc:
    A = noisy
    B = randomize(noisy, std_dev=16.0)
    C = project_A_to_B(A, B, estimator, block_size)
    mrc.set_data(C)
    mrc.data

