import numpy as np
import matplotlib.pyplot as plt

class LichtenbergFigure:
    def __init__(self, grid, bounds):
        self.grid = grid
        self.Rc = grid.shape[0] // 2
        self.bounds = bounds
        self.lower, self.upper = self.bounds

        self.scale = 1
        self.rot = 0

    def sample(self, n):
        flat = self.grid.flatten()
        indices = np.random.choice(len(flat), p=flat/np.sum(flat), size=n)
        coords = []
        for idx in indices:
            coord = np.unravel_index(idx, self.grid.shape)
            # 2d rotate the coord about the center (Rc, Rc)
            coord = np.array(coord) - self.Rc
            coord = np.dot(coord, [[np.cos(self.rot), -np.sin(self.rot)], [np.sin(self.rot), np.cos(self.rot)]]) + self.Rc
            # scale the coord about (Rc, Rc)
            coord = (coord - self.Rc) * self.scale + self.Rc
            coords.append([self.lower + (self.upper - self.lower) * coord[i] / (self.grid.shape[i] - 1) for i in range(len(coord))])
        
        return coords
    
    def rand_transform(self):
        self.scale = np.random.uniform(0.01, 1)
        self.rot = np.random.uniform(0, 2*np.pi)

    def copy(self, ref=1.):
        copy = LichtenbergFigure(self.grid.copy(), self.bounds)
        copy.scale = self.scale * ref
        copy.rot = self.rot
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
        self.experience = {"it": [], "fitness": [], "coords": []}
        
    def optimize(self, J, n_iter: int, pop: int):
        if self.M == 2:
            grid = np.load(self.filename)
        
        lf = LichtenbergFigure(grid, J.bounds())
        trigger = J.center()
        best_coords = trigger
        best_fitness = J.evaluate(*trigger)

        for it in range(n_iter):
            lf.rand_transform()
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

            trigger = best_coords
        
        return best_coords
    
    def plot_convergence(self):
        if not self.experience['it']:
            return
        
        plt.plot(self.experience["it"], self.experience["fitness"])
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.show()
