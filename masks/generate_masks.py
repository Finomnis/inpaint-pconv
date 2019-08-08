#!/usr/bin/env python3

import argparse
import os
import random
import math

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from scipy.ndimage import morphology
from tqdm import tqdm


class NBodySim:

    class SimState:

        def __init__(self, p_x, p_y, v_x, v_y):
            self.p_x = p_x
            self.p_y = p_y
            self.v_x = v_x
            self.v_y = v_y

        def __add__(self, other):

            p_x = [d1 + d2 for d1, d2 in zip(self.p_x, other.p_x)]
            p_y = [d1 + d2 for d1, d2 in zip(self.p_y, other.p_y)]
            v_x = [d1 + d2 for d1, d2 in zip(self.v_x, other.v_x)]
            v_y = [d1 + d2 for d1, d2 in zip(self.v_y, other.v_y)]

            return NBodySim.SimState(p_x, p_y, v_x, v_y)

        def __radd__(self, other):
            return self.__add__(other)

        def __mul__(self, val):
            p_x = [d*val for d in self.p_x]
            p_y = [d*val for d in self.p_y]
            v_x = [d*val for d in self.v_x]
            v_y = [d*val for d in self.v_y]

            return NBodySim.SimState(p_x, p_y, v_x, v_y)

        def __rmul__(self, other):
            return self.__mul__(other)

        def __str__(self):
            result = "NBodySim.SimState():\n"
            result += '   p_x: ' + str(self.p_x) + "\n"
            result += '   p_y: ' + str(self.p_y) + "\n"
            result += '   v_x: ' + str(self.v_x) + "\n"
            result += '   v_y: ' + str(self.v_y) + "\n"
            return result

    def __init__(self, num):

        self.state = NBodySim.SimState([], [], [], [])
        self.log_x = list()
        self.log_y = list()
        self.mass = list()
        self.num = num

        for _ in range(num):
            self.state.p_x.append(random.uniform(0, 1))
            self.state.p_y.append(random.uniform(0, 1))
            self.state.v_x.append(random.uniform(-0.05, 0.05))
            self.state.v_y.append(random.uniform(-0.05, 0.05))
            self.mass.append(random.uniform(0.01, 0.015))
            self.log_x.append([self.state.p_x[-1]])
            self.log_y.append([self.state.p_y[-1]])

    def get_state(self):
        return self.state

    def update_state(self, state):
        self.state = state
        for i in range(self.num):
            self.log_x[i].append(self.state.p_x[i])
            self.log_y[i].append(self.state.p_y[i])

    def compute_derivative(self, state):

        acc_x = list()
        acc_y = list()

        for i in range(self.num):
            p_x = state.p_x[i]
            p_y = state.p_y[i]
            mass = self.mass[i]

            f_x_total = 0
            f_y_total = 0
            for j in range(self.num):
                if i == j:
                    continue
                p_other_x = state.p_x[j]
                p_other_y = state.p_y[j]
                dist_x = p_other_x - p_x
                dist_y = p_other_y - p_y
                dist = math.sqrt(dist_x*dist_x+dist_y*dist_y)
                if dist == 0:
                    continue

                dir_x = dist_x / dist
                dir_y = dist_y / dist

                if dist < 0.02:
                    dist = 0.02

                force = mass * self.mass[j] / (dist*dist)

                force_x = force*dir_x
                force_y = force*dir_y

                f_x_total += force_x
                f_y_total += force_y

            acc_x.append(f_x_total / mass)
            acc_y.append(f_y_total / mass)

        return NBodySim.SimState(state.v_x.copy(), state.v_y.copy(), acc_x, acc_y)

    def step_euler(self, dt):
        state_old = self.get_state()
        deriv = self.compute_derivative(state_old)
        new_state = state_old + deriv * dt

        self.update_state(new_state)

    def step_runge_kutta(self, dt):
        state_old = self.get_state()

        k1 = dt * self.compute_derivative(state_old)
        k2 = dt * self.compute_derivative(state_old + k1 * 0.5)
        k3 = dt * self.compute_derivative(state_old + k2 * 0.5)
        k4 = dt * self.compute_derivative(state_old + k3)

        new_state = state_old + (1.0/6.0) * (k1 + 2*k2 + 2*k3 + k4)

        self.update_state(new_state)

    def step(self, dt):
        # self.stepEuler(dt)
        self.step_runge_kutta(dt)

    def __repr__(self):
        result = "NBodySim():\n"
        result += '   p_x: ' + str(self.state.p_x) + "\n"
        result += '   p_y: ' + str(self.state.p_y) + "\n"
        result += '   v_x: ' + str(self.state.v_x) + "\n"
        result += '   v_y: ' + str(self.state.v_y) + "\n"
        return result


def generate_random_stencil(max_radius):

    line_radius = random.randint(1, max_radius)
    stencil_type = random.choice(('square', 'circle', 'diamond'))

    stencil = np.ones((2*line_radius+1, 2*line_radius+1), dtype=np.uint8)

    if stencil_type == 'circle':
        for y in range(stencil.shape[0]):
            p_y = (y - line_radius) / line_radius
            for x in range(stencil.shape[1]):
                p_x = (x - line_radius) / line_radius
                if p_x*p_x+p_y*p_y >= 1:
                    stencil[y, x] = 0

    elif stencil_type == 'diamond':
        for y in range(stencil.shape[0]):
            p_y = (y - line_radius) / line_radius
            for x in range(stencil.shape[1]):
                p_x = (x - line_radius) / line_radius
                if abs(abs(p_x) + abs(p_y)) >= 1:
                    stencil[y, x] = 0

    return stencil


def generate_random_paths(size_x, size_y):

    while True:
        n_body_sim = NBodySim(5)
        for i in range(1000):
            n_body_sim.step(0.01)

        # plt.figure('Sim')
        # for log_x, log_y in zip(n_body_sim.log_x, n_body_sim.log_y):
        #     plt.plot(log_x, log_y, '+-')
        #     plt.plot(log_x[-1:], log_y[-1:], 'o')
        # axes = plt.axes()
        # axes.set_xlim([0, 1])
        # axes.set_ylim([0, 1])

        for x_log, y_log in zip(n_body_sim.log_x, n_body_sim.log_y):
            img = Image.new('1', size=(size_x, size_y), color=0)
            draw = ImageDraw.Draw(img)

            line_data = [(x*size_x, y*size_y) for x, y in zip(x_log, y_log)]

            draw.line(line_data, fill=1)
            yield img


def generate_random_dots(size_x, size_y):
    img = Image.new('1', size=(size_x, size_y), color=0)
    pixels = img.load()

    num_dots = random.randint(1, size_x*size_y // 2000)
    for _ in range(num_dots):
        x = random.randrange(size_x)
        y = random.randrange(size_y)
        pixels[x, y] = 1

    return img


def main():
    parser = argparse.ArgumentParser(description='Creates a number of masks.')
    parser.add_argument('--out-folder', default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'generated'),
                        help='The output folder', )
    parser.add_argument('--samples', type=int, default=10000, help='The number of masks to generate')
    parser.add_argument('--out-size', nargs=2, type=int, default=[512, 512], metavar=('SIZE_X', 'SIZE_Y'),
                        help='The size of the generated masks')
    args = parser.parse_args()
    # print(args)

    if not os.path.isdir(args.out_folder):
        os.makedirs(args.out_folder)

    file_format_string = "{:0" + str(len(str(args.samples-1))) + "d}.png"

    random_path_gen = iter(generate_random_paths(args.out_size[0], args.out_size[1]))
    for i in tqdm(range(args.samples)):
        stencil = generate_random_stencil(min(args.out_size)//20)
        random_path = next(random_path_gen)
        mask = morphology.binary_dilation(random_path, structure=stencil)
        if random.choice((True, False)):
            random_dots = generate_random_dots(args.out_size[0], args.out_size[1])
            random_dots_stencil = generate_random_stencil(min(args.out_size)//20)
            random_dots_mask = morphology.binary_dilation(random_dots, structure=random_dots_stencil)
            mask = np.logical_xor(mask, random_dots_mask)
            if random.choice((True, False)):
                random_dots = generate_random_dots(args.out_size[0], args.out_size[1])
                random_dots_stencil = generate_random_stencil(min(args.out_size)//200)
                random_dots_mask = morphology.binary_dilation(random_dots, structure=random_dots_stencil)
                mask = np.logical_xor(mask, random_dots_mask)

        mask = Image.fromarray(255-mask.astype(np.uint8)*255, mode='L')

        if False:
            plt.figure('Stencil')
            plt.subplot(1, 3, 1)
            plt.imshow(stencil)
            plt.subplot(1, 3, 2)
            plt.imshow(random_path)
            plt.subplot(1, 3, 3)
            plt.imshow(mask)
            plt.show()
            exit(1)

        file_name = os.path.join(args.out_folder, file_format_string.format(i))
        mask.save(file_name)


if __name__ == "__main__":
    main()
