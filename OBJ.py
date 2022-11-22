import numpy


class Obj(object):
    def __init__(self, filename):
        with open(filename) as f:
            self.lines = f.read().splitlines()
            self.vertices = []
            self.faces = []

            for line in self.lines:
                if len(line) > 1:
                    # print(line)
                    prefix, value = line.split(" ", 1)

                    if prefix == "v":
                        self.vertices.append(list(map(float, value.split(" "))))
                    if prefix == "f":
                        self.faces.append(
                            [
                                list(map(int, face.split("/")))
                                for face in value.split(" ")
                            ]
                        )

        self.vertices = numpy.array(self.vertices, dtype=numpy.float32)
        self.faces = numpy.array(self.faces, dtype=numpy.float32)
