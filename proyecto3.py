import pygame
from math import pi, cos, sin, atan2
import time
import random


SKY = (50, 100, 200)
GROUND = (164, 160, 147, 255)

# Inicializamos Pygame
pygame.init()
# Definimos tamaño de pantalla
screen = pygame.display.set_mode((1000, 600), pygame.DOUBLEBUF | pygame.OPENGL)
screen = pygame.display.set_mode((1000, 600))

# Paredes del Juego
wall1 = pygame.image.load("./WallValo.png").convert()
wall2 = pygame.image.load("./ValoWall2.jpg").convert()
wall3 = pygame.image.load("./B_SITE.png").convert()
wall4 = pygame.image.load("./A_SITE.png").convert()
wall5 = pygame.image.load("./C_SITE.png").convert()
wall6 = pygame.image.load("./PreRoung.png").convert()

# Enemigos
enemy1 = pygame.image.load("./sprite1.png").convert()

# Jugador
weapon = pygame.image.load("./prime.png").convert()

# Texturas de las paredes traducidas al txt
textures = {"1": wall1, "2": wall2, "3": wall3, "4": wall4, "5": wall5, "6": wall6}

# Posicion de los enemigos
enemies = [{"x": 400, "y": 100, "texture": enemy1}]

# Clase RayCarter
class Raycaster:
    def __init__(self, screen):
        _, _, self.width, self.height = screen.get_rect()
        self.screen = screen
        self.blocksize = 50
        self.map = []
        self.zbuffer = [-float("inf") for z in range(0, 500)]
        self.player = {
            "x": self.blocksize + self.blocksize / 2,
            "y": self.blocksize + self.blocksize / 2,
            "fov": int(pi / 3),
            "a": int(pi / 3),
        }

    # Coloca los pixeles dentro de la pantalla
    def point(self, x, y, c=None):
        screen.set_at((x, y), c)

    # Dibujar minimapa en esquina superior derecha
    def minimap(self, x, y, texture, size):
        for cx in range(x, x + size):
            for cy in range(y, y + size):
                tx = int((cx - x) * 12.8)
                ty = int((cy - y) * 12.8)
                c = texture.get_at((tx, ty))
                self.point(cx, cy, c)

    # Dibujar al jugador
    def draw_player(self, xi, yi, element, w=500, h=500):
        for x in range(xi, xi + w):
            for y in range(yi, yi + h):
                tx = int((x - xi) * 0.125)
                ty = int((y - yi) * 0.125)
                c = element.get_at((tx, ty))
                if c != (255, 255, 255, 255):
                    self.point(x, y, c)

    # Cargar el mapa
    def load_map(self, filename):
        with open(filename) as f:
            for line in f.readlines():
                self.map.append(list(line))

    # Funcion de Cast Ray
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
                tx = int(maxhit * 2.56)
                return d, self.map[j][i], tx
            d += 1

    # Funcion draw_stake
    def draw_stake(self, x, h, tx, texture):
        h_half = h / 2
        start = int(250 - h_half)
        end = int(250 + h_half)
        end_start_pro = 128 / (end - start)
        for y in range(start, end):
            ty = int((y - start) * end_start_pro)
            c = texture.get_at((tx, ty))
            self.point(x, y, c)

    # Funcion para dibujar el sprite del enemigo
    def draw_sprite(self, sprite):
        sprite_a = atan2(
            (sprite["y"] - self.player["y"]), (sprite["x"] - self.player["x"])
        )
        sprite_d = (
            (self.player["x"] - sprite["x"]) ** 2
            + (self.player["y"] - sprite["y"]) ** 2
        ) ** 0.5
        sprite_size_half = int(250 / sprite_d * 70)
        sprite_size = sprite_size_half * 2
        sprite_x = int(500 + (sprite_a - self.player["a"]) * 500 - sprite_size_half)
        sprite_y = int(250 - sprite_size_half)

        sprite_size_pro = 128 / sprite_size
        for x in range(sprite_x, sprite_x + sprite_size):
            for y in range(sprite_y, sprite_y + sprite_size):
                if 500 < x < 1000 and self.zbuffer[x - 500] <= sprite_d:
                    tx = int((x - sprite_x) * sprite_size_pro)
                    ty = int((y - sprite_y) * sprite_size_pro)
                    c = sprite["texture"].get_at((tx, ty))
                    if c != (152, 0, 136, 255):
                        self.point(x, y, c)
                        self.zbuffer[x - 500] = sprite_d

    def render(self):
        for i in range(0, 1000):
            try:
                a = self.player["a"] - self.player["fov"] / 2 + (i * 0.00105)
                d, m, tx = self.cast_ray(a)
                x = i
                h = (500 / (d * cos(a - self.player["a"]))) * 50
                self.draw_stake(x, h, tx, textures[m])
            except:
                self.player["x"] = 70
                self.player["y"] = 70
                self.lost()
        # Dibujar enemigos
        for enemy in enemies:
            self.point(enemy["x"], enemy["y"], (0, 0, 0))
            self.draw_sprite(enemy)

        # Minimap
        for x in range(0, 100, 10):
            for y in range(0, 100, 10):
                i = int(x * 0.1)
                j = int(y * 0.1)
                if self.map[j][i] != " ":
                    y = 0 + y
                    x1 = 900 + x
                    self.minimap(x1, y, textures[self.map[j][i]], 10)

        # Player
        self.point(
            int(self.player["x"] * 0.2) + 900,
            int(self.player["y"] * 0.2) + 500,
            (90, 30, 5),
        )

        # Dibujar Arma
        self.draw_player(500, 244, weapon)

    # create a rectangular object for the
    # text surface object
    # https://www.geeksforgeeks.org/python-display-text-to-pygame-window/
    def text(self, text, font):
        textSurface = font.render(text, True, (255, 255, 255))
        return textSurface, textSurface.get_rect()

    # Pantalla Inicial
    def home_screen(self):
        home = True

        while home:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    exit(0)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_z:
                        home = False
                        self.start()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_i:
                        home = False
                        self.controls()

            # Referencia
            # https://www.youtube.com/watch?v=eS5G5b3Ruvc
            # https://coderslegacy.com/python/pygame-font/
            screen.fill((100, 25, 40))
            TextSurf, TextRect = self.text(
                "WELCOME TO VALORANT", pygame.font.Font("Valorant_Font.ttf", 70)
            )
            TextRect.center = (500, 250)
            screen.blit(TextSurf, TextRect)
            TextSurf, TextRect = self.text(
                "PRESS Z TO START OR PRESS I TO CHECK CONTROLS",
                pygame.font.Font("Valorant_Font.ttf", 30),
            )
            TextRect.center = (500, 350)
            screen.blit(TextSurf, TextRect)
            TextSurf, TextRect = self.text(
                "ESC PARA SALIR", pygame.font.Font("Valorant_Font.ttf", 20)
            )
            TextRect.center = (500, 450)
            screen.blit(TextSurf, TextRect)
            pygame.display.update()

    # Funcion de lost
    def lost(self):
        # Se reproduce el sonido cuando pierde
        lose_sound = pygame.mixer.Sound("./Defeat.mp3")
        lose_sound.set_volume(0.1)
        pygame.mixer.Sound.play(lose_sound)
        pygame.mixer.music.stop()

        home = True
        while home:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    exit(0)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        home = False
                        self.home_screen()

            # Referencia
            # https://www.youtube.com/watch?v=eS5G5b3Ruvc
            # https://coderslegacy.com/python/pygame-font/
            screen.fill((255, 0, 0))
            TextSurf, TextRect = self.text(
                "DEFEAT", pygame.font.Font("Valorant_Font.ttf", 70)
            )
            TextRect.center = (500, 250)
            screen.blit(TextSurf, TextRect)
            TextSurf, TextRect = self.text(
                "YOU'VE BEEN HEADSHOTED OR CRASHED INTO A WALL",
                pygame.font.Font("Valorant_Font.ttf", 30),
            )
            TextRect.center = (500, 350)
            screen.blit(TextSurf, TextRect)
            TextSurf, TextRect = self.text(
                "LOOK FOR ANOTHER ANGLE", pygame.font.Font("Valorant_Font.ttf", 20)
            )
            TextRect.center = (500, 400)
            screen.blit(TextSurf, TextRect)
            TextSurf, TextRect = self.text(
                "ESC PARA SALIR", pygame.font.Font("Valorant_Font.ttf", 20)
            )
            TextRect.center = (500, 450)
            screen.blit(TextSurf, TextRect)
            pygame.display.update()

    # Funcion cuando gana el juego
    def win(self):
        # Se reproduce sonido de victoria
        win_sound = pygame.mixer.Sound("./Win.mp3")
        win_sound.set_volume(0.1)
        pygame.mixer.Sound.play(win_sound)
        pygame.mixer.music.stop()

        home = True
        while home:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    exit(0)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        home = False
                        self.home_screen()

            # Referencia
            # https://www.youtube.com/watch?v=eS5G5b3Ruvc
            # https://coderslegacy.com/python/pygame-font/
            screen.fill((127, 255, 147))
            TextSurf, TextRect = self.text(
                "YOU WON THE DUEL", pygame.font.Font("Valorant_Font.ttf", 70)
            )
            TextRect.center = (500, 250)
            screen.blit(TextSurf, TextRect)
            TextSurf, TextRect = self.text(
                "ESC PARA SALIR", pygame.font.Font("Valorant_Font.ttf", 20)
            )
            TextRect.center = (500, 350)
            screen.blit(TextSurf, TextRect)
            pygame.display.update()

    # Pantalla con los controles del juego
    def controls(self):
        home = True

        while home:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    exit(0)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_z:
                        home = False
                        self.start()

            # Referencia
            # https://www.youtube.com/watch?v=eS5G5b3Ruvc
            # https://coderslegacy.com/python/pygame-font/
            screen.fill((100, 25, 40))
            TextSurf, TextRect = self.text(
                "Move Right => D", pygame.font.Font("Valorant_Font.ttf", 30)
            )
            TextRect.center = (500, 100)
            screen.blit(TextSurf, TextRect)
            TextSurf, TextRect = self.text(
                "Move Left => A", pygame.font.Font("Valorant_Font.ttf", 30)
            )
            TextRect.center = (500, 150)
            screen.blit(TextSurf, TextRect)
            TextSurf, TextRect = self.text(
                "Move Up => W", pygame.font.Font("Valorant_Font.ttf", 30)
            )
            TextRect.center = (500, 200)
            screen.blit(TextSurf, TextRect)
            TextSurf, TextRect = self.text(
                "Move Down => S", pygame.font.Font("Valorant_Font.ttf", 30)
            )
            TextRect.center = (500, 250)
            screen.blit(TextSurf, TextRect)
            TextSurf, TextRect = self.text(
                "Play Music => O", pygame.font.Font("Valorant_Font.ttf", 30)
            )
            TextRect.center = (500, 300)
            screen.blit(TextSurf, TextRect)
            TextSurf, TextRect = self.text(
                "Use the mouse scroll to look left and right",
                pygame.font.Font("Valorant_Font.ttf", 30),
            )
            TextRect.center = (500, 350)
            screen.blit(TextSurf, TextRect)
            TextSurf, TextRect = self.text(
                "PRESS Z TO START", pygame.font.Font("Valorant_Font.ttf", 70)
            )
            TextRect.center = (500, 450)
            screen.blit(TextSurf, TextRect)
            TextSurf, TextRect = self.text(
                "ESC PARA SALIR", pygame.font.Font("Valorant_Font.ttf", 20)
            )
            TextRect.center = (500, 500)
            screen.blit(TextSurf, TextRect)
            pygame.display.update()

    # Sonido del Juego
    def sound(self):
        pygame.mixer.music.load("./ValoMusic.mp3")
        pygame.mixer.music.set_volume(0.1)
        pygame.mixer.music.play(-1)

    # contador de FPS
    def fpsCounter(self):
        fuente = pygame.font.Font(None, 25)
        text_de_salida = "FPS: " + str(round(clock.get_fps(), 2))
        text = fuente.render(text_de_salida, True, (255, 255, 255))
        return text

    # Funcion que inicializa el juego y tiene el movimiento del pesonaje
    def start(self):
        paused = False
        running = True
        d = 10
        while running:
            screen.fill(SKY)
            screen.fill(GROUND, (0, r.height / 2.5, r.width, r.height))
            screen.blit(self.fpsCounter(), [0, 0])
            if (r.player["x"] >= 300 and r.player["x"] <= 500) and (
                r.player["y"] >= 200 and r.player["y"] <= 250
            ):
                self.win()
            if (r.player["x"] >= 300 and r.player["x"] <= 500) and (
                r.player["y"] >= 100 and r.player["y"] <= 250
            ):
                self.lost()
            r.render()
            pygame.display.flip()
            clock.tick(60)

            for e in pygame.event.get():
                if e.type == pygame.QUIT or (
                    e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE
                ):
                    running = False
                    exit(0)
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_a:
                        r.player["a"] -= pi / 10
                    if e.key == pygame.K_d:
                        r.player["a"] += pi / 10
                    if e.key == pygame.K_w:
                        r.player["x"] += int(d * cos(r.player["a"]))
                        r.player["y"] += int(d * sin(r.player["a"]))
                    if e.key == pygame.K_s:
                        r.player["x"] -= int(d * cos(r.player["a"]))
                        r.player["y"] -= int(d * sin(r.player["a"]))
                    if e.key == pygame.K_o:
                        r.sound()
                if e.type == pygame.MOUSEBUTTONUP or e.type == pygame.MOUSEBUTTONDOWN:
                    if e.button == 4:
                        r.player["a"] -= pi / 10
                    if e.button == 5:
                        r.player["a"] += pi / 10


# Se inicializa el raycaster
r = Raycaster(screen)

# Se carga el mapa
r.load_map("./map.txt")

# Se usa clock para los fps
clock = pygame.time.Clock()

# empieza la pantalla de inicio
r.home_screen()
