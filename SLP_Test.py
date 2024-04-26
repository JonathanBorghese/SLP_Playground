import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

for d in range(10):

    weights = np.fromfile("weights/w" + str(d) + "_normalized_weights.txt")

    weights = np.reshape(weights, (28, 28))

    #min = np.amin(weights)
    #max = np.amax(weights)

    min = np.percentile(weights, 5)
    max = np.percentile(weights, 95)

    norm = mpl.colors.Normalize(vmin = min, vmax = max)

    fig, ax = plt.subplots()
    im = ax.imshow(weights, norm=norm, cmap='magma')

    plt.tight_layout()
    plt.show()