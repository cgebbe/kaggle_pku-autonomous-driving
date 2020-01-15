import numpy as np
import matplotlib.pyplot as plt

cars = [{'v': 120, 'u': 500, 'z': 10},
        {'v': 100, 'u': 200, 'z': 20},
        ]

height = 320
width = 1024
confidence = np.zeros((height, width))
us, vs = np.meshgrid(np.arange(width), np.arange(height))
for car in cars:
    u, v = car['u'], car['v']
    # choose sigma s.t. sigma = 2px at 10m and 8x downsample (40x128)
    sigma = 10 * 10 / car['z']
    heatmap = np.exp(- ((us - u) ** 2 + (vs - v) ** 2) / (2.0 * sigma))
    confidence += heatmap

mask_low = confidence < 1E-12
confidence[mask_low] = 0

plt.imshow(confidence)
plt.show()
print("=== Finished")