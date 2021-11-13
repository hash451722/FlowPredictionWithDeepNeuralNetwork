import pathlib
import tkinter

import numpy as np
import matplotlib.pyplot as plt



def load_data(file_path):

    dat = np.load(file_path)
    print(dat.shape)
    print(file_path.stem)
    return dat, file_path.stem


def draw_data(dat=None, filename=None):

    fig, axs = plt.subplots(3, 3, figsize=(7, 7))

    axs[0, 0].imshow(dat[0])  # mask
    axs[0, 0].set_title('mask', fontsize=10)
    axs[1, 0].imshow(dat[1])  # fs x
    axs[1, 0].set_title('freestream x', fontsize=10)
    axs[2, 0].imshow(dat[2])  # fs y
    axs[2, 0].set_title('freestream y', fontsize=10)

    axs[0, 1].imshow(dat[3])  # pressure
    axs[0, 1].set_title('pressure', fontsize=10)
    axs[1, 1].imshow(dat[4])  # vx
    axs[1, 1].set_title('velocity x', fontsize=10)
    axs[2, 1].imshow(dat[5])  # vy
    axs[2, 1].set_title('velocity y', fontsize=10)

    fig.suptitle(filename)
    plt.show()



if __name__ == '__main__':

    data_path = pathlib.Path(__file__).parent.joinpath("airfoil.npy")

    dat, file_name = load_data(data_path)

    draw_data(dat, file_name)