from manim import *
from scipy import integrate
from datetime import datetime
import numdifftools as nd
import matplotlib.pyplot as plt
from metodos_numericos import *


class DipoloElectrico2D(VMobject): # noqa

    def __init__(self, angle,
                 pos=np.array([6e-3, 6e-3, 0]),
                 t_start=0,
                 t_end=30,
                 ele=140e-9,
                 q=1.6e-19,
                 m=1e-16,
                 fields=(),
                 pos_draw=np.array([0, 0, 0]),
                 longitud_escala=(40e-9, 140e-9),
                 fps=30,
                 set_time_interval=True,
                 solucionador='Radau',
                 resolver_angulo=True,
                 resolver_desplazamiento=True,
                 momento_inercia_tipo='esfera',
                 logging=False,
                 plots=False):

        super().__init__()

        star_time = datetime.now()
        if logging:
            print(f"Inicio de ejecución: {star_time} [h:mm:ss.μs]\n")

        self.angle = angle  # con respecto al eje horizontal del sistema coordenado de referencia
        self.pos = pos
        self.t_start = t_start
        self.t_end = t_end
        self.ele = ele
        self.q1 = q
        self.q2 = -q
        self.m = m
        self.fields = fields
        self.pos_draw = pos_draw
        self.longitud_escala = longitud_escala
        self.fps = int(fps)
        self.set_time_interval = set_time_interval
        self.solucionador = solucionador  # 'Radau', 'RK45', 'myRK45'
        self.resolver_angulo = resolver_angulo
        self.resolver_desplazamiento = resolver_desplazamiento
        self.momento_inercia_tipo = momento_inercia_tipo  # 'esfera', 'dos_masas', 'barra'
        self.logging = logging
        self.plots = plots
        self.solution = None
        self.count_steps = 0

        self.dE = nd.Jacobian(self.campo_en_punto)  # gradiente o jacobiano, es lo mismo ya que es una función vectorial
        self.momento_inercia_method = self.decide_momento_inercia_method()  # definir qué método de momento de inercia
        # definir qué métodos usar para calcular el desplazamiento y el ángulo
        self.traslacion_dipolo_method = self.traslacion_dipolo if self.resolver_desplazamiento \
            else self.traslacion_dipolo_mock
        self.angulo_dipolo_method = self.angulo_dipolo if self.resolver_angulo else self.angulo_dipolo_mock

        self.create()
        self.calculate_dynamics()

        end_time = datetime.now()

        if self.logging:
            print(f"Tiempo de ejecución: {end_time - star_time} [h:mm:ss.μs]\n\n")

    def create(self):

        longitud2 = self.escalar([self.ele], self.longitud_escala, [0.15, 0.8])[0]
        stroke_w = self.escalar([self.ele], self.longitud_escala, [4, 9])[0]
        scaledot = self.escalar([self.ele], self.longitud_escala, [1, 3])[0]

        linea = Line(LEFT * longitud2, RIGHT * longitud2, stroke_width=stroke_w)    # 0.15 - 0.8, 4 - 9

        polo_positivo = Dot(color=RED).scale(scaledot).move_to(linea.get_end())     # 1 - 3
        polo_negativo = Dot(color=BLUE).scale(scaledot).move_to(linea.get_start())  # 1 - 3

        linea.add(polo_positivo, polo_negativo)

        self.add(linea)

        self.move_to(self.pos_draw)

    def calculate_dynamics(self):

        theta = self.angle  # ángulo inicial
        omega = 0.0  # velocidad angular inicial

        p, l = self.momento_dipolar(theta, self.pos)

        I = self.momento_inercia_esfera_solida(self.m, l)

        E = self.campo_en_punto(self.pos)

        if self.logging:
            # cálculos para ver como son los números y sus órdenes de magnitud (al inicio de la solución numérica)

            tau = np.cross(p, E)[2]  # el valor de tau queda en la coordenada 'z' y con el signo correcto
            dE_r = self.dE(self.pos)
            ax = (1 / self.m) * (p[0] * dE_r[0][0] + p[1] * dE_r[0][1] + p[2] * dE_r[0][2])

            print("theta =", theta)
            print("self.pos[0] =", self.pos[0])
            print("l =", l)
            print("p =", p)
            print("I =", I)
            print("E =", E)
            print('tau =', tau)
            print("dE_r =", dE_r)
            print("alpha =", tau / I)
            print("a_x =", ax)
            print("------------")
            print("cantidad puntos =", self.t_end * self.fps + 2)
            print("[theta, omega] = ", [theta, omega])

        if self.solucionador == 'myRK45':
            self.solution = rk45(self.movimiento_dipolo,
                                 self.t_start,
                                 np.array([self.pos[0], self.pos[1], self.pos[2], 0, 0, 0, theta, omega]),
                                 self.t_end * self.fps + 2,
                                 1/self.fps)
        else:
            self.solution = integrate.solve_ivp(self.movimiento_dipolo,
                                                [self.t_start, self.t_end],
                                                [self.pos[0], self.pos[1], self.pos[2], 0, 0, 0, theta, omega],
                                                method=self.solucionador,
                                                t_eval=np.linspace(self.t_start, self.t_end, self.t_end * self.fps + 2)
                                                if self.set_time_interval else None
                                                )

        if self.plots:
            self.show_plots()

    def movimiento_dipolo(self, t, y):

        r = y[0:6]        # componentes de la traslación
        theta = y[6:8]    # componentes de la rotación

        pos = r[0:3]      # posición actual
        angle = theta[0]  # ángulo actual

        self.count_steps = self.count_steps + 1  # conteo de pasos realizados (llamadas a este método)

        p, l = self.momento_dipolar(angle, pos)  # momento dipolar en la posición actual

        y_traslacion = self.traslacion_dipolo_method(t, r, p)      # calcular cambio en el desplazamiento

        y_angulo = self.angulo_dipolo_method(t, theta, pos, p, l)  # calcular cambio en el ángulo

        return np.concatenate((y_traslacion, y_angulo))

    def angulo_dipolo(self, t, theta, pos, p, l):

        # componentes
        x, v = theta  # x: theta, v: omega (ángulo y velocidad angular)

        I = self.momento_inercia_method(self.m, l)  # en la posición actual

        E = self.campo_en_punto(pos)  # en la posición actual

        # tau = np.linalg.norm(p) * np.linalg.norm(E) * np.sin(theta)
        tau = np.cross(p, E)[2]  # el valor de tau queda en la coordenada z y con el signo correcto

        a = tau / I if I != 0 else 0  # el signo está incluido en la variable 'tau'

        return [v, a]  # velocidad (omega, primera derivada) y aceleración (alpha, segunda derivada)

    def angulo_dipolo_mock(self, t, theta, pos, p, l):
        return [theta[1], 0.0]

    def traslacion_dipolo(self, t, r, p):

        # componentes
        x, y, z, vx, vy, vz = r    # posición  y velocidades correspondientes por coordenada

        dE_r = self.dE([x, y, z])  # gradiente en la posición actual

        ax = (1 / self.m) * (p[0] * dE_r[0][0] + p[1] * dE_r[0][1] + p[2] * dE_r[0][2])
        ay = (1 / self.m) * (p[0] * dE_r[1][0] + p[1] * dE_r[1][1] + p[2] * dE_r[1][2])
        az = (1 / self.m) * (p[0] * dE_r[2][0] + p[1] * dE_r[2][1] + p[2] * dE_r[2][2])

        return [vx, vy, vz, ax, ay, az]  # velocidades y aceleraciones por coordenada

    def traslacion_dipolo_mock(self, t, r, p):
        return [r[3], r[4], r[5], 0.0, 0.0, 0.0]

    def momento_dipolar(self, angle, center_pos):

        # considerando que el centro de 'ele' está en 'center_pos'
        r_pos = np.array(
            [(self.ele / 2) * np.cos(angle), (self.ele / 2) * np.sin(angle), 0]) + center_pos    # carga positva
        r_neg = np.array(
            [-(self.ele / 2) * np.cos(angle), -(self.ele / 2) * np.sin(angle), 0]) + center_pos  # carga negativa

        l = r_pos - r_neg
        p = l * self.q1

        return p, l  # momento dipolar y vector 'l' que define la separación de las cargas

    def campo_en_punto(self, r):

        total_field = np.array([0., 0., 0.])
        for field in self.fields:
            total_field += field.campo_real(r)

        return total_field

    def show_plots(self):

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)

        fig.suptitle(fr"Dinámica del dipolo con $\theta_0=${self.angle:.4f} [rad], "
                     fr"$x_0=${self.pos[0]:.1e} [m], $y_0=${self.pos[1]:.1e} [m]")

        if self.solucionador == 'myRK45':
            ax1.plot(self.solution[0], [row[0] for row in self.solution[1]], label="x")
            ax1.plot(self.solution[0], [row[1] for row in self.solution[1]], label="y")
        else:
            ax1.plot(self.solution.t, self.solution.y[0], label="x")
            ax1.plot(self.solution.t, self.solution.y[1], label="y")
        ax1.set(ylabel='Distancia [m]')
        ax1.legend()

        if self.solucionador == 'myRK45':
            ax2.plot(self.solution[0], [row[6] for row in self.solution[1]])
        else:
            ax2.plot(self.solution.t, self.solution.y[6])
        ax2.set(xlabel='Tiempo [s]', ylabel='Ángulo [rad]')

        plt.show()

    def zoom_plot(self, start_r, end_r, tipo_plot='angulo'):

        if tipo_plot == 'angulo':
            if self.solucionador == 'myRK45':
                plt.plot(self.solution[0][start_r:end_r], [row[6] for row in self.solution[1]][start_r:end_r])
            else:
                plt.plot(self.solution.t[start_r:end_r], self.solution.y[6][start_r:end_r])
            plt.xlabel('Tiempo [s]')
            plt.ylabel('Ángulo [rad]')
            plt.show()
        elif tipo_plot == 'distancia':
            if self.solucionador == 'myRK45':
                plt.plot(self.solution[0][start_r:end_r], [row[0] for row in self.solution[1]][start_r:end_r],
                         label="x")
                plt.plot(self.solution[0][start_r:end_r], [row[1] for row in self.solution[1]][start_r:end_r],
                         label="y")
            else:
                plt.plot(self.solution.t[start_r:end_r], self.solution.y[0][start_r:end_r], label="x")
                plt.plot(self.solution.t[start_r:end_r], self.solution.y[1][start_r:end_r], label="y")
            plt.xlabel('Tiempo [s]')
            plt.ylabel('Distancia [m]')
            plt.legend()
            plt.show()

    def decide_momento_inercia_method(self):
        if self.momento_inercia_tipo == 'esfera':
            return self.momento_inercia_esfera_solida
        elif self.momento_inercia_tipo == 'dos_masas':
            return self.momento_inercia_dos_masas
        elif self.momento_inercia_tipo == 'barra':
            return self.momento_inercia_barra

    @staticmethod
    def momento_inercia_barra(m, l):

        return m * np.linalg.norm(l) ** 2 / 12

    @staticmethod
    def momento_inercia_dos_masas(m, l):

        # 'm' es la masa total, entonces m1 = m2 = m/2
        # usa la masa reducida = m1m2/(m1+m2) = (m^2/4) / (m) = m/4

        return m * np.linalg.norm(l) ** 2 / 4

    @staticmethod
    def momento_inercia_esfera_solida(m, l):

        # aquí se considera el radio (no el diámetro)

        return 2 * m * (np.linalg.norm(l) / 2) ** 2 / 5

    @staticmethod
    def escalar(values, actual_bounds, desired_bounds):
        return [
            desired_bounds[0] + (x - actual_bounds[0]) * (desired_bounds[1] - desired_bounds[0])
            / (actual_bounds[1] - actual_bounds[0])
            for x in values
        ]

class Virus2D(DipoloElectrico2D):  # noqa

    config.assets_dir = "../img"

    def create(self):

        diametro = self.escalar([self.ele], self.longitud_escala, [0.2, 0.8])[0]

        virus = SVGMobject("coronavirus").scale(diametro)

        canvas = VGroup(*virus)

        self.add(canvas)

        self.move_to(self.pos_draw)

class VirusEsfera2D(DipoloElectrico2D):  # noqa

    # con fines de pruebas, solo dibuja una circunferencia con un diametro para notar su rotación
    def create(self):

        diametro = self.escalar([self.ele], self.longitud_escala, [0.2, 0.8])[0]

        virus = Circle(radius=diametro/2)

        linea = Line(-diametro/2, diametro/2)

        virus.add(linea)

        self.add(virus)

        self.move_to(self.pos_draw)
