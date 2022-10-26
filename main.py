from cyglfw3 import *
from OpenGL.GL import *

glfw.Init()

window = glfW.CreateWindow(800, 600, 'opengl')

while not glfw.WindowShouldClose(window):

    glfw.PollEvents()

glfw.Terminate()