from matplotlib import pyplot as plt


def draw_2darray(array, filename):
    plt.clf()
    plt.imshow(array)
    plt.colorbar()
    plt.savefig(filename)