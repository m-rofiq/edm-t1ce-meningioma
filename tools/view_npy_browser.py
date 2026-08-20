import os
import numpy as np
import matplotlib.pyplot as plt

# =========================
# SET FOLDER
# =========================

FOLDER = r"./data/dataset_5fold_final_v4/holdout_test/T1ce"

# =========================
# LOAD FILES
# =========================

files = sorted([f for f in os.listdir(FOLDER) if f.endswith(".npy")])

idx = 0


def show_image():

    global idx

    file = files[idx]
    path = os.path.join(FOLDER, file)

    arr = np.load(path)

    if arr.ndim == 3:
        img = arr[1]   # center slice
    else:
        img = arr

    plt.clf()

    plt.imshow(img, cmap="gray")
    plt.title(f"{idx+1}/{len(files)} : {file}")
    plt.axis("off")

    plt.draw()


def on_key(event):

    global idx

    if event.key == "n":

        idx = min(idx + 1, len(files)-1)
        show_image()

    elif event.key == "b":

        idx = max(idx - 1, 0)
        show_image()

    elif event.key == "q":

        plt.close()


# =========================
# RUN VIEWER
# =========================

fig = plt.figure()

fig.canvas.mpl_connect("key_press_event", on_key)

show_image()

plt.show()