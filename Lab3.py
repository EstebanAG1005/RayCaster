# Esteban Aldana Guerra 20591
# Graficas por Computadora
# Laboratorio 3 Game of Life

import pygame
from OpenGL.GL import *
import copy
import random

# Se inicializa pygame
pygame.init()

# Se define una escala para los pixeles
scale = 10

# se define el tamaño de la pantalla
width = 51
height = 51

screen = pygame.display.set_mode((width*scale, height*scale),pygame.OPENGL | pygame.DOUBLEBUF)

# Funcion para los pixeles
def pixel(x, y, color):
    glEnable(GL_SCISSOR_TEST)
    glScissor(x*scale, y*scale, 1*scale, 1*scale)
    glClearColor(color[0], color[1], color[2], 1.0)
    glClear(GL_COLOR_BUFFER_BIT)
    glDisable(GL_SCISSOR_TEST)

# Array vacio de pixeles que vamos a almacenar 
pixels = []

# Se usa para poder recorrer los pixeles que tenemos
for x in range(1, width+1):
    temp = []
    for y in range(1, height+1):
        xy = 0
        temp.append(xy)
        # print(xy)
        y = +1
    pixels.append(temp)
    x = +1

# Funcion la cual define la logica del Juego 
def logic(x, y, pixels):
    count = 0
    # Recorre los diferentes pixeles que definimos
    if x > 0 and x < 50 and y > 0 and y < 50:
        if pixels[x-1][y+1] == 1:
            count += 1
        if pixels[x][y+1] == 1:
            count += 1
        if pixels[x+1][y+1] == 1:
            count += 1
        if pixels[x-1][y] == 1:
            count += 1
        if pixels[x+1][y] == 1:
            count += 1
        if pixels[x-1][y-1] == 1:
            count += 1
        if pixels[x][y-1] == 1:
            count += 1
        if pixels[x+1][y-1] == 1:
            count += 1

    return count

# Funcion para dibujar los diferentes pixeles dentro de la pantalla
def draw():
    for x in range(len(pixels)):
        for y in range(len(pixels)):
            # Color del Pixel Blanco 
            if pixels[x][y] == 1:
                pixel(x, y, (1, 1, 1))
            else:
                # Color del pixel Background tipo azul XD
                pixel(x, y, (0.1, 0.3, 0.6))

# Funcion de Update para dar la logica al juego y poder mostrar los diferentes cambios
def update():
    # Toma la ultima captura de los pixeles
    last_pixels = copy.deepcopy(pixels)
    for x in range(width):
        for y in range(height):
            if pixels[x][y] == 1:
                # underpopulation
                if logic(x, y, last_pixels) < 2:
                    pixels[x][y] = 0
                # survival
                if logic(x, y, last_pixels) > 3:
                    pixels[x][y] = 0
                # overpopulation
                if logic(x, y, last_pixels) == 2 or logic(x, y, last_pixels) == 3:
                    pixels[x][y] = 1

            else:
                # reproduction
                if logic(x, y, last_pixels) == 3:
                    pixels[x][y] = 1
    draw()

# retorna el size de la pantalla
size = (width * height)/3
x = width
y = height

# Generacion de pixeles
while size > 0:
    random_y = random.randint(1, y-5)
    pixels[25][random_y] = 1
    size -= 1

running = True
while running:
    # clean
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)
    # paint
    update()
    # flip
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False