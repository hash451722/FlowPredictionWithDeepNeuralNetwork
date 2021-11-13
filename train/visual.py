import pathlib
import tkinter

import numpy as np
import matplotlib.pyplot as plt



def load_data(file_path):

    dat = np.load(file_path)
    print(dat.shape)
    print(file_path.stem)
    return dat, file_path.stem



def draw_data(dat):
    print("Visual")
    title = [["mask", "freestream x", "freestream y"],
             ["pressure", "velocity x", "velocity y"],
             ["pressure", "velocity x", "velocity y"]]

    fig, axs = plt.subplots(3, 3, figsize=(7, 7))

    k = 0
    for i in range(3):
        for j in range(3):
            axs[j,i].imshow(dat[k])
            axs[j,i].set_title(title[i][j], fontsize=10)
            k += 1

    plt.show()



if __name__ == '__main__':

    data_path = pathlib.Path(__file__).parent.joinpath("airfoil.npy")

    dat, file_name = load_data(data_path)

    draw_data(dat, file_name)