import pygame
import random
from math import pi, cos, sin, atan2
import time

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

SKY = (50, 100, 200)
GROUND = (200, 200, 100)

colors = [(0, 20, 10), (4, 40, 63), (0, 91, 82), (219, 242, 38), (21, 42, 138)]

walls = {
    "1": pygame.image.load("./caja.png"),
    "2": pygame.image.load("./caja.png"),
    "3": pygame.image.load("./wall1.png"),
    "4": pygame.image.load("./wall1.png"),
    "5": pygame.image.load("./wall1.png"),
}

sprite1 = pygame.image.load("./sprite1.png")


enemies = [
    {"x": 100, "y": 200, "texture": pygame.image.load("./sprite1.png")},
    {"x": 280, "y": 190, "texture": pygame.image.load("./sprite1.png")},
    {"x": 225, "y": 340, "texture": pygame.image.load("./sprite1.png")},
    {"x": 220, "y": 425, "texture": pygame.image.load("./sprite1.png")},
    {"x": 320, "y": 420, "texture": pygame.image.load("./sprite1.png")},
]


class Raycaster(object):
    def __init__(self, screen):
        self.screen = screen
        _, _, self.width, self.height = screen.get_rect()
        self.blocksize = 50
        self.map = []
        self.zbuffer = [-float("inf") for z in range(0, 500)]
        self.player = {
            "x": self.blocksize + self.blocksize / 2,
            "y": self.blocksize + self.blocksize / 2,
            "fov": int(pi / 3),
            "a": int(pi / 3),
        }

    def fpsCounter(self):
        fuente = pygame.font.Font(None, 25)
        texto_de_salida = "FPS: " + str(round(clock.get_fps(), 2))
        texto = fuente.render(texto_de_salida, True, WHITE)
        return texto

    def point(self, x, y, c=WHITE):
        # colocar pixel de game of life
        self.screen.set_at((x, y), c)

    def block(self, x, y, wall):
        for i in range(x, x + self.blocksize):
            for j in range(y, y + self.blocksize):
                tx = int((i - x) * 128 / self.blocksize)
                ty = int((j - y) * 128 / self.blocksize)
                c = wall.get_at((tx, ty))
                self.point(i, j, c)

    def load_map(self, filename):
        with open(filename) as f:
            for line in f.readlines():
                self.map.append(list(line))

    def draw_stake(self, x, h, tx, c):
        start_y = int(self.height / 2 - h / 2)
        end_y = int(self.height / 2 + h / 2)

        height = end_y - start_y

        for y in range(start_y, end_y):
            ty = int((y - start_y) * 128 / height)
            color = walls[c].get_at((tx, ty))
            self.point(x, y, color)

    def cast_ray(self, a):
        d = 0
        ox = self.player["x"]
        oy = self.player["y"]

        while True:
            x = int(ox + d * cos(a))
            y = int(oy + d * sin(a))

            i = int(x / self.blocksize)
            j = int(y / self.blocksize)

            if self.map[j][i] != " ":
                hitx = x - i * self.blocksize
                hity = y - j * self.blocksize

                if 1 < hitx < self.blocksize - 1:
                    maxhit = hitx
                else:
                    maxhit = hity

                tx = int(maxhit * 128 / self.blocksize)
                return d, self.map[j][i], tx

            self.point(x, y)
            d += 5

    def draw_map(self):
        for x in range(0, 500, self.blocksize):
            for y in range(0, 500, self.blocksize):
                i = int(x / self.blocksize)
                j = int(y / self.blocksize)
                if self.map[j][i] != " ":
                    self.block(x, y, walls[self.map[j][i]])

    def draw_player(self):
        self.point(int(self.player["x"]), int(self.player["y"]))

    def draw_sprite(self, sprite):
        sprite_a = atan2(
            sprite["y"] - self.player["y"], sprite["x"] - self.player["x"]
        )  # why atan2? https://stackoverflow.com/a/12011762
        sprite_d = (
            (self.player["x"] - sprite["x"]) ** 2
            + (self.player["y"] - sprite["y"]) ** 2
        ) ** 0.5
        sprite_size = (500 / sprite_d) * 70

        sprite_x = (
            500
            + (sprite_a - self.player["a"]) * 500 / self.player["fov"]
            + 250
            - sprite_size / 2
        )
        sprite_y = 250 - sprite_size / 2

        sprite_x = int(sprite_x)
        sprite_y = int(sprite_y)
        sprite_size = int(sprite_size)

        for x in range(sprite_x, sprite_x + sprite_size):
            for y in range(sprite_y, sprite_y + sprite_size):
                if 500 < x < 1000 and self.zbuffer[x - 500] >= sprite_d:
                    tx = int((x - sprite_x) * 128 / sprite_size)
                    ty = int((y - sprite_y) * 128 / sprite_size)
                    c = sprite["texture"].get_at((tx, ty))
                    if c != (152, 0, 136, 255):
                        self.point(x, y, c)
                        self.zbuffer[x - 500] = sprite_d

    def render(self):
        self.draw_map()
        self.draw_player()

        density = 100

        # minimap
        for i in range(0, density):
            a = (
                self.player["a"]
                - self.player["fov"] / 2
                + self.player["fov"] * i / density
            )
            d, c, _ = self.cast_ray(a)

        # separador
        for i in range(0, 500):
            self.point(499, i)
            self.point(500, i)
            self.point(501, i)

        # draw in 3d
        for i in range(0, int(self.width / 2)):
            a = (
                self.player["a"]
                - self.player["fov"] / 2
                + self.player["fov"] * i / (self.width / 2)
            )
            d, c, tx = self.cast_ray(a)

            x = int(self.width / 2 + i)
            h = self.height / (d * cos(a - self.player["a"])) * 100

            self.draw_stake(x, h, tx, c)
            self.zbuffer[i] = d

        for enemy in enemies:
            self.point(enemy["x"], enemy["y"], (0, 0, 0))
            self.draw_sprite(enemy)


pygame.init()
# screen = pygame.display.set_mode((1000, 500), pygame.FULLSCREEN)
screen = pygame.display.set_mode((1000, 500))
r = Raycaster(screen)
r.load_map("./map.txt")
clock = pygame.time.Clock()

running = True
while running:
    screen.blit(r.fpsCounter(), [600, 550])
    screen.fill(BLACK, (0, 0, r.width / 2, r.width / 2))
    screen.fill(SKY, (r.width / 2, 0, r.width, r.height / 2))
    screen.fill(GROUND, (r.width / 2, r.height / 2, r.width, r.height / 2))
    r.render()

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_s:
                r.player["a"] += pi / 10
            if event.key == pygame.K_w:
                r.player["a"] -= pi / 10

            if event.key == pygame.K_UP:
                r.player["x"] += 10
            if event.key == pygame.K_DOWN:
                r.player["x"] -= 10
            if event.key == pygame.K_LEFT:
                r.player["y"] -= 10
            if event.key == pygame.K_RIGHT:
                r.player["y"] += 10
