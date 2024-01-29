# Influencia de campos eléctricos y magnéticos sobre dipolos eléctricos y magnéticos

Código utilizado en la tesis *Influencia de campos eléctricos y magnéticos sobre dipolos eléctricos y magnéticos* por **Claudia Zendejas-Morales**

Se generaron simulaciones numéricas para entender y visualizar el comportamiento de dipolos bajo la influencia de campos eléctricos y magnéticos. Unas visualizaciones se hicieron con [Matplotlib](https://matplotlib.org/) y otras con [Manim](https://github.com/3b1b/manim).

## Visualizaciones con Matplotlib

Se encuentran en la carpeta [`sim_matplotlib`](https://github.com/clausia/dipoles-electromagnetic-fields/tree/main/codigo/sim_matplotlib) y el código fue escrito dentro de Jupyter notebooks, para su fácil asociación con los resultados. Los archivos principales son:

- `ejemplos-campos-electricos.ipynb` que contiene el código utilizado para mostrar cómo se ven los campos eléctricos bajo distintas circunstancias, las imágenes generadas aquí fueron utilizadas principalmente en el capítulo 3
- `ejemplos-campos-magneticos.ipynb` contiene el código utilizado para mostrar cómo se ven los campos magnéticos, las imágenes fueron utilizadas principalmente en el capítulo 5

## Visualizaciones con Manim

Se encuentran en la carpeta [`sim_manim`](https://github.com/clausia/dipoles-electromagnetic-fields/tree/main/codigo/sim_manim) y el código fue escrito tanto en Jupyter notebooks como en archivos `.py`; a continuación la descripción de algunos de los archivos:

- `dipolo_electrico.py`: contiene la escena de Manim que muestra el movimiento de dipolos eléctricos bajo la influencia de campos, al crear este objeto, se pueden específicar multiples parámetros para que la simulación pueda ser personalizada como se describe en la sección 8.3
- `generadores_campo.py`: define una clase base para especificar objetos generadores de campo,. asi mismo contienen clases concretas con generadores como placas paralelas o cargas puntuales
- `metodos_numericos.py`: aquí se puede encontrar la implementación del método Runge-Kutta-Fehlberg o RK45, como se describe en la sección 8.1

Los distintos notebooks contienen código para generar videos con animaciones hechas en Manim. Algunos videos pueden encontrarse aqui:

- Dipolo bajo un campo generado por una carga puntual [video](https://youtu.be/tOT0nUgboJA), el dipolo visto como un virus SARS-CoV-2 [video](https://youtu.be/gj11WIBRaSQ)
- Dipolo bajo la influencia del campo generado por varias cargas puntuales [video](https://youtu.be/J-M8embV6d8)
- Dipolos bajo la influencia de un campo generado por un par de placas palalelas [video](https://youtu.be/HWn7V46Tte0)
- Dipolos (con forma de virus) bajos la influencia del campo generado por varias cargas puntuales [video](https://youtu.be/Vhx-LJVa0_o), aquí no se calcula la oscilación pero si el desplazamiento, debido a que resulta un cálculo muy pesado y lento, la idea es mostrar el desplazamiento, nótese que los dipolos corresponden al valor del dipolo del virus solamente, sin influencia de moléculas de agua como se describe en la sección 7.7




