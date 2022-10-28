import pygame
from OpenGL.GL import *
import numpy as np


pygame.init()


scale = 7

dimensiony = 70
dimensionx = 70

screen = pygame.display.set_mode(
    (dimensionx*scale, dimensiony*scale),
    pygame.OPENGL | pygame.DOUBLEBUF
)


glClearColor(0.0, 1.0, 0.0, 1.0)

cells = np.zeros((dimensiony, dimensionx))
pattern = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0],
                    [1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]);


def update(surface, cells, cellsize):
    nxt = np.zeros((cells.shape[0], cells.shape[1]))

    for r, c in np.ndindex(cells.shape):
        num_alive = np.sum(cells[r-1:r+2, c-1:c+2]) - cells[r, c]

        if cells[r, c] == 1 and num_alive < 2 or num_alive > 3:
            col = col_about_to_die
        elif (cells[r, c] == 1 and 2 <= num_alive <= 3) or (cells[r, c] == 0 and num_alive == 3):
            nxt[r, c] = 1
            col = col_alive

        col = col if cells[r, c] == 1 else col_background

    return nxt

def pixel(x,y,color):
    glEnable(GL_SCISSOR_TEST)
    glScissor(x*scale, y*scale, 1*scale, 1*scale)
    glClearColor(color[0],color[1],color[2],1.0)
    glClear(GL_COLOR_BUFFER_BIT)
    glDisable(GL_SCISSOR_TEST)


running = True
while running:
    # clean
    glClearColor(0.1,0.8,0.2,1.0)
    glClear(GL_COLOR_BUFFER_BIT)

    # paint
    for y,row in enumerate(pattern):
        for x,cell in enumerate(row):
            if cell == 1:
                pixel(x, y, (1.0,0.0,0.0))


    #Update segun el array inicial
    
    #Ejecuta y muestra el nuevo array



    # flip
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False