from abc import ABC, abstractmethod
from manim import *


class GeneradorCampo(ABC, VMobject):

    @abstractmethod
    def _create(self):
        pass

    @abstractmethod
    def campo_draw(self, r):
        pass

    @abstractmethod
    def campo_real(self, r):
        pass


class Placas2D(GeneradorCampo):  # noqa

    def __init__(self,
                 ancho=0.5,
                 alto=7.5,
                 E=500e3,
                 E_draw=1):

        super().__init__()

        self.ancho = ancho
        self.alto = alto
        self.E = E
        self.E_draw = E_draw
        self._create()

    def _create(self):
        placa_positiva = Rectangle(width=self.ancho, height=self.alto, color=RED_B) \
            .set_fill(color=RED_E, opacity=1).move_to(LEFT * (7.0 - self.ancho))
        placa_negativa = Rectangle(width=self.ancho, height=self.alto, color=BLUE) \
            .set_fill(color=BLUE_E, opacity=1).move_to(RIGHT * (7.0 - self.ancho))

        self.add(placa_positiva, placa_negativa)

    def _campo(self, p, E_mag):

        E = 0

        half_s = ScreenRectangle().height  # altura de la mitad de la pantalla

        # hay campo solo en el espacio entre las placas
        if (LEFT[0] * (7.0 - self.ancho)) < p[0] < (RIGHT[0] * (7.0 - self.ancho)) \
                and -half_s + (half_s - self.alto / 2) < p[1] < half_s - (half_s - self.alto / 2):
            E = E_mag

        return np.array([
            E,  # x
            0,  # y
            0   # z
        ])

    def campo_draw(self, p):
        return self._campo(p, self.E_draw)

    def campo_real(self, p):
        return self._campo(p, self.E)


class CargaPuntual2D(GeneradorCampo):  # noqa

    def __init__(self,
                 pos=np.array([0, 0, 0]),
                 q=1.6e-19,
                 q_draw=1,
                 radio=0.15):

        super().__init__()

        self.pos = pos
        self.q = -np.abs(q) if q < 0 or q_draw < 0 else q
        self.q_draw = -np.abs(q_draw) if q < 0 or q_draw < 0 else q_draw
        self.radio = radio
        self.k = 8.99e9

        self._create()

    def _create(self):
        carga = Circle(radius=self.radio, color=RED_B if self.q_draw > 0 else BLUE_B) \
            .set_fill(color=RED_E if self.q_draw > 0 else BLUE_E, opacity=1).move_to(self.pos)

        self.add(carga)

    def _campo(self, r, q, k):
        x, y, z = r

        # validar en dónde esta la carga y su interior, porque ahi el campo se hará cero
        # ya que no podrá pasar nunca a la parte de adentro de un dipolo (o un virus)
        if (x - self.pos[0]) ** 2 + (y - self.pos[1]) ** 2 + (z - self.pos[2]) ** 2 < self.radio ** 2:
            return np.array([0, 0, 0])

        denominador = (((x - self.pos[0]) ** 2 + (y - self.pos[1]) ** 2 + (z - self.pos[2]) ** 2) ** (3 / 2))

        return np.array([
            k * (x - self.pos[0]) * q / denominador if denominador != 0 else 0,  # x coord
            k * (y - self.pos[1]) * q / denominador if denominador != 0 else 0,  # y coord
            k * (z - self.pos[2]) * q / denominador if denominador != 0 else 0   # z coord
        ])

    def campo_draw(self, r):
        return self._campo(r, self.q_draw, 1)

    def campo_real(self, r):
        return self._campo(r, self.q, self.k)
