import tkinter
import math

import gridlogic as gl

window = tkinter.Tk()
window.title("Gridworld")

def create_grid(block_width, block_height, block_size, matrix):
    matrix_copy = matrix.copy().flatten()
    for i in range (block_width*block_height):
        canva = tkinter.Canvas(width=block_size, height=block_size)
        canva.create_rectangle(0, 0, block_size, block_size, fill="white", outline="black")
        canva.create_text(block_size/2, block_size/2, text=matrix_copy[i])
        canva.grid(row=int(math.floor(i/block_height)), column=i%block_width)

gridMatrix = gl.init_grid_matrix(5,5)
create_grid(len(gridMatrix[0]), len(gridMatrix), 50, gridMatrix)

window.mainloop()