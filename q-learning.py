import numpy as np
import cv2
import time
import random
import matplotlib.pyplot as plt

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# CONEXIÓN CON SIMULADOR
class Coppelia():
    def __init__(self):
        client = RemoteAPIClient()
        self.sim = client.getObject('sim')

    def start_simulation(self):
        self.default_idle_fps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)
        self.sim.startSimulation()

    def stop_simulation(self):
        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            time.sleep(0.1)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, self.default_idle_fps)

    def is_running(self):
        return self.sim.getSimulationState() != self.sim.simulation_stopped


# ROBOT I/O
class P3DX():
    num_sonar = 16
    sonar_max = 1.0

    def __init__(self, sim, robot_id, use_camera=True):
        self.sim = sim
        self.left_motor = self.sim.getObject(f'/{robot_id}/leftMotor')
        self.right_motor = self.sim.getObject(f'/{robot_id}/rightMotor')
        if use_camera:
            self.camera = self.sim.getObject(f'/{robot_id}/camera')

        # posición inicial del robot para reset
        self.robot_handle = self.sim.getObject(f'/{robot_id}')
        self.initial_position = self.sim.getObjectPosition(self.robot_handle, -1)
        self.initial_orientation = self.sim.getObjectOrientation(self.robot_handle, -1) 

    def get_image(self):
        img, resX, resY = self.sim.getVisionSensorCharImage(self.camera)
        img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
        img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
        return img

    def set_speed(self, left_speed, right_speed):
        self.sim.setJointTargetVelocity(self.left_motor, left_speed)
        self.sim.setJointTargetVelocity(self.right_motor, right_speed)

    def reset_position(self):
        self.sim.setObjectPosition(self.robot_handle, -1, self.initial_position)
        self.sim.setObjectOrientation(self.robot_handle, -1, self.initial_orientation)


# DIMENSIONES Q-TABLE
NUM_STATES = 7
NUM_ACTIONS = 7

Q = np.zeros((NUM_STATES, NUM_ACTIONS))

# HIPERPARÁMETROS
EPSILON_START = 0.1   # exploración: epsilon-decay (definir start=0 para epsilon constante)
EPSILON_MIN = 0
EPSILON_DECAY = 0.98

ALPHA_START = 0.4     # tasa de aprendizaje: alpha-decay (definir start=0 para alpha constante)
ALPHA_MIN = 0.05
ALPHA_DECAY = 0.98

GAMMA = 0.9           # factor de descuento


def process_image(image):
    roi = image[100:255, 0:255] 
    gray_img = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    _, proc_img = cv2.threshold(blurred_img, 60, 255, cv2.THRESH_BINARY_INV)

    _, width = proc_img.shape
    
    M = cv2.moments(proc_img)

    return width, M


def get_state_from_image(image):
    width, M = process_image(image)

    if M['m00'] == 0:
        return -1, -1 # no está la línea en la imagen
    
    x = int(M['m10'] / M['m00'])

    # desplazamiento con respecto al centro
    center = width // 2
    offset = x - center
    max_offset = width // 2

    # normalización
    offset_norm = offset / max_offset

    # discretización
    state = min((NUM_STATES - 1), int((offset_norm + 1) / (2 / NUM_STATES)))

    return state, offset_norm
    

def choose_action(state, epsilon):
    if np.random.rand() < epsilon:
        return random.randint(0, NUM_ACTIONS - 1)
    
    return np.argmax(Q[state])


def apply_action(robot, action):
    base_speed = 2
    turn_range = 1

    # proportion en rango [-1, 1]
    proportion = (action / (NUM_ACTIONS - 1)) * 2 - 1

    left_speed = base_speed - turn_range * proportion
    right_speed = base_speed + turn_range * proportion

    robot.set_speed(left_speed, right_speed)


def get_reward(state, offset):
    if state == -1:
        return -10
    
    return 1 - 2 * abs(offset)


def check_completed(sim, goal_sensor):
    result, _, _, _, _ = sim.readProximitySensor(goal_sensor)
    return result


EPISODES = 100

def main(args=None):
    print(f'\n--- Iniciando entrenamiento con {NUM_STATES} estados y {NUM_ACTIONS} acciones ---')
    coppelia = Coppelia()
    robot = P3DX(coppelia.sim, 'PioneerP3DX', use_camera=True)
    goal_sensor = coppelia.sim.getObject('/proximitySensor')
    coppelia.start_simulation()

    reward_per_episode = []
    offset_per_episode = []

    start_time = time.time()

    try:
        for episode in range(EPISODES):
            epsilon = max(EPSILON_MIN, EPSILON_START * (EPSILON_DECAY ** episode))
            alpha = max(ALPHA_MIN, ALPHA_START * (ALPHA_DECAY ** episode))

            total_reward = 0
            total_offset = 0
            steps = 0
            delay = 500

            while coppelia.is_running():
                image = robot.get_image()
                state, _ = get_state_from_image(image)
                action = choose_action(state, epsilon)
                apply_action(robot, action)

                new_image = robot.get_image()
                new_state, offset = get_state_from_image(new_image)
                reward = get_reward(new_state, offset)
                total_reward += reward
                total_offset += abs(offset)

                # Actualizar Q-table
                Q[state, action] += alpha * (reward + GAMMA * np.max(Q[new_state]) - Q[state, action])

                steps += 1

                # Reiniciar si pierde la línea
                if state == -1:
                    break

                # terminar si se completa el circuito
                if steps > delay and check_completed(coppelia.sim, goal_sensor):
                    break

            offset_per_episode.append(total_offset / steps)
            reward_per_episode.append(total_reward)

            print(f'--- Episodio: {episode} --- Alpha: {round(alpha, 2)}--- Epsilon: {round(epsilon, 2)} ---')

            robot.reset_position()

            if not coppelia.is_running():
                break

        finish_time = time.time()

        # visualización de resultados
        print('\n--- ENTRENAMIENTO COMPLETADO ---')
        print(f'Tiempo: {round(finish_time - start_time, 2)}s')
        print(f'Episodios: {len(reward_per_episode)}')

        plt.plot(reward_per_episode)
        plt.title(f'Recompensa total por episodio para {NUM_STATES} estados y {NUM_ACTIONS} acciones')
        plt.xlabel('Episodio')
        plt.ylabel('Recompensa total')
        plt.grid(True)
        plt.show()

        plt.plot(offset_per_episode)
        plt.title(f'Desplazamiento medio por episodio para {NUM_STATES} estados y {NUM_ACTIONS} acciones')
        plt.xlabel('Episodio')
        plt.ylabel('Desplazamiento')
        plt.grid(True)
        plt.show()


    finally:
        coppelia.stop_simulation()


if __name__ == '__main__':
    main()