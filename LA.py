import numpy as np
import matplotlib.pyplot as plt

class LichtenbergFigure:
    def __init__(self, grid, center, bounds):
        self.grid = grid
        self.Rc = grid.shape[0] // 2
        self.center = center
        self.bounds = bounds

        self.trigger = self.center
        self.lower, self.upper = self.bounds

        self.scale = 1
        self.rot = 0

    def sample(self, n):
        flat = self.grid.flatten()
        indices = np.random.choice(len(flat), p=flat/np.sum(flat), size=n)
        coords = []
        factor = (self.upper - self.lower) / (self.Rc * 2)
        for idx in indices:
            coord = np.unravel_index(idx, self.grid.shape)
            # 2d rotate the coord about the center (Rc, Rc)
            coord = np.array(coord) - self.Rc
            coord = np.dot(coord, [[np.cos(self.rot), -np.sin(self.rot)], [np.sin(self.rot), np.cos(self.rot)]]) + self.Rc
            # scale the coord about (Rc, Rc)
            coord = (coord - self.Rc) * self.scale + self.Rc
            coord = [factor * coord[i] + self.lower + self.trigger[i] - self.center[i] for i in range(len(coord))]
            coords.append(coord)
        
        return coords
    
    def rand_transform(self, trigger):
        self.scale = np.random.uniform(0.01, 1)
        self.rot = np.random.uniform(0, 2*np.pi)
        self.trigger = trigger

    def copy(self, ref=1.):
        copy = LichtenbergFigure(self.grid.copy(), self.center, self.bounds)
        copy.scale = self.scale * ref
        copy.rot = self.rot
        copy.trigger = self.trigger
        return copy


class LichtenbergAlgorithm:
    # based on "Lichtenberg algorithm: A novel hybrid physics-based meta-heuristic for global optimization" by Pereira et al.

    def __init__(self, M: int, **kwargs):
        self.M = M

        if self.M == 2:
            if 'filename' not in kwargs:
                raise ValueError("Filename must be provided for M=2")
            self.filename = kwargs['filename']
        
        self.ref = kwargs.get('ref', 0)
        self.experience = {"it": [], "fitness": [], "coords": [], "samples": []}
        
    def optimize(self, J, n_iter: int, pop: int):
        if self.M == 2:
            grid = np.load(self.filename)
        
        trigger = J.center()
        best_coords = trigger
        best_fitness = J.evaluate(*trigger)

        lf = LichtenbergFigure(grid, trigger, J.bounds())

        for it in range(n_iter):
            lf.rand_transform(trigger)
            if not self.ref:
                samples = lf.sample(pop)
            else:
                lf2 = lf.copy(self.ref)
                global_pop = lf.sample(int(pop * 0.4))
                local_pop = lf2.sample(int(pop * 0.6))
                samples = global_pop + local_pop
            
            sample_fitnesses = [(J.evaluate(*s), s) for s in samples]
            min_fitness, min_sample = min(sample_fitnesses)

            if min_fitness < best_fitness:
                best_fitness = min_fitness
                best_coords = min_sample
                
            self.experience["it"].append(it)
            self.experience["fitness"].append(best_fitness)
            self.experience["coords"].append(best_coords)
            self.experience["samples"].extend(samples)

            trigger = best_coords
        
        return best_coords
    
    def plot_convergence(self):
        if not self.experience['it']:
            return
        
        plt.plot(self.experience["it"], self.experience["fitness"])
        plt.xlabel("Iteration")
        plt.ylabel("$f_{min}(X)$")
        plt.title("Convergence")
        plt.show()

    def plot_historical_search(self):
        if not self.experience['it']:
            return

        coords = np.array(self.experience["coords"])
        samples = np.array(self.experience["samples"])

        plt.scatter(samples[:, 0], samples[:, 1], c='b', marker='.')
        plt.plot(coords[:-1, 0], coords[:-1, 1], c='r')
        plt.scatter(coords[-1, 0], coords[-1, 1], c='r', marker='x')
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.title('Historical Search')
        plt.show()
