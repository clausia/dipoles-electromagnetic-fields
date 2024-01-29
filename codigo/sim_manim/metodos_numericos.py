

def rk45(f, t0, y0, nmax, h):
    """
    Método Runge–Kutta–Fehlberg (o método de Fehlberg)

    método de orden O(h^4) con un estimador de error de orden O(h^5)

    [https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Fehlberg]

    :param f:    función
    :param t0:   tiempo inicial
    :param y0:   valor inicial, un solo valor o un vector (arreglo) de valores (para múltiples variables a resolver)
    :param nmax: número de iteraciones
    :param h:    tamaño del paso
    :return:     Lista con dos listas: valores del tiempo, y valores de 'y' calculados
    """

    y = y0
    t = t0
    t_values = [t]
    y_values = [y]

    for i in range(1, nmax + 1):
        k1 = h * f(t, y)
        k2 = h * f(t + h / 4, y + k1 / 4)
        k3 = h * f(t + h * 3 / 8, y + k1 * 3 / 32 + k2 * 9 / 32)
        k4 = h * f(t + h * 12 / 13, y + k1 * 1932 / 2197 - k2 * 7200 / 2197 + k3 * 7296 / 2197)
        k5 = h * f(t + h, y + k1 * 439 / 216 - k2 * 8 + k3 * 3680 / 513 - k4 * 845 / 4104)
        k6 = h * f(t + h * 1 / 2, y - k1 * 8 / 27 + k2 * 2 - k3 * 3544 / 2565 + k4 * 1859 / 4104 - k5 * 11 / 40)
        k = 16 / 135 * k1 + 0 * k2 + 6656 / 12825 * k3 + 28561 / 56430 * k4 - 9 / 50 * k5 + 2 / 55 * k6
        y = y + k
        t = t + h
        t_values.append(t)
        y_values.append(y)

    return [t_values, y_values]
