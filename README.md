# Q-learning_TFG

### Título
Seguimiento de líneas con técnicas de aprendizaje por refuerzo en robótica móvil

### Autor
Eduardo López Delmás

### Descripción
El trabajo trata del diseño e implementación de un controlador para un robot móvil, cuya función es seguir una línea dibujada sobre el suelo. Para lograr este objetivo se implementan en Python técnicas de aprendizaje por refuerzo, concretamente el Q-learning, en combinación con técnicas de visión por computadora con OpenCV. La simulación de la interacción física del robot con su entorno se realiza a través del simulador CoppeliaSim.

### Contenido
- **q-learning.py:** Contiene la lógica para comunicarse con CoppeliaSim, implementar un agente de aprendizaje por refuerzo con Q-learning, entrenar al agente y visualizar los resultados.
- **pioneer-tfg.ttt:** Contiene la escena en la que se simula el proceso físico de entrenamiento del agente.
