import numpy as np
import matplotlib.pyplot as plt

class Cell:
    def __init__(self, i, j, potential=0.):
        self.i = i
        self.j = j
        self.phi = potential

    def get_coords(self):
        return self.i, self.j    

class DielectricBreakdownModel:
    # based on "Fast Simulation of Laplacian Growth" by Kim et al.
    # fuse breakdown instead of dielectric breakdown simulated for effectively same result

    def __init__(self, eta=5., dim=100, h=1, seed=None):
        self.eta = eta
        # size of one grid block
        self.R1 = h/2

        # 0 = empty, 1 = filled
        self.dim = dim
        self.grid = np.zeros((2*dim+1, 2*dim+1))
        if not seed:
            seed = [dim, dim]
        
        self.pattern = np.array([seed])
        self.candidates = []
        self.environment = []
        self.hit_edge = False
        self._generate_neighbors()

        seed = Cell(seed[0], seed[1])
        self.add_cell(seed)

    def _generate_neighbors(self):
        self.neighbors = {}
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                self.neighbors[(i, j)] = [(i+1, j), (i-1, j), (i, j+1), (i, j-1),
                                     (i+1, j+1), (i+1, j-1), (i-1, j+1), (i-1, j-1)]     
                self.neighbors[(i, j)] = [n for n in self.neighbors[(i, j)] if 0 < n[0] < self.grid.shape[0] and 0 < n[1] < self.grid.shape[1]]

    def grow_pattern(self):
        cell = self.choose_candidate()
        self.add_cell(cell)
        self.update_potentials(cell)

    def choose_candidate(self):
        phi_values = np.array([c.phi for c in self.candidates])  
        max_phi = np.max(phi_values)
        min_phi = np.min(phi_values)

        phi_eta = ((phi_values - min_phi) / (max_phi - min_phi)) ** self.eta
        phi_eta_sum = np.sum(phi_eta)
        probabilities = phi_eta / phi_eta_sum
        
        idx = np.random.choice(len(self.candidates), p=probabilities)
        return self.candidates.pop(idx)

    def simulate(self, steps=None):
        if not steps:
            steps = 0
            while not self.hit_edge:
                self.grow_pattern()
                steps += 1
        else:
            for _ in range(int(steps)):
                self.grow_pattern()
        return steps

    def add_cell(self, cell):
        self.grid[cell.i, cell.j] = 1
        self.pattern = np.append(self.pattern, [[cell.i, cell.j]], axis=0)

        neighbors = self.neighbors[(cell.i, cell.j)]
        if len(neighbors) < 8:
            self.hit_edge = True
        
        for (ni, nj) in neighbors:
            # only allow for neighbors in neutral space
            if self.grid[ni, nj] != 0:
                continue
            self.grid[ni, nj] = -1
            self.candidates.append(Cell(ni, nj, self.calculate_spawn_potential(ni, nj)))
        
    def calculate_spawn_potential(self, i, j):
        # equation 10
        pattern_distances = np.sqrt((i - self.pattern[:, 0]) ** 2 + (j - self.pattern[:, 1]) ** 2)
        neighbor_potential = np.sum(1 - self.R1 / pattern_distances)

        environment_potential = np.sum([c.phi / np.sqrt((i - c.i) ** 2 + (j - c.j) ** 2) for c in self.environment])
        return neighbor_potential + environment_potential

    def update_potentials(self, cell):
        # equation 11
        candidate_coords = np.array([[c.i, c.j] for c in self.candidates])
        distances = np.sqrt((candidate_coords[:, 0] - cell.i) ** 2 + (candidate_coords[:, 1] - cell.j) ** 2)
        delta_potentials = 1 - self.R1 / distances
        for c, delta in zip(self.candidates, delta_potentials):
            c.phi += delta

    def show(self, show_environment=False):
        img = self.grid
        if not show_environment:
            img[np.where(img!=1)] = 0
            img = 1 - img
            plt.imshow(img, cmap='gray')
        else:
            img[np.where(img > 1)] = 2
            img[np.where(img < -1)] = -2
            plt.imshow(img)
        plt.axis('off')
        plt.show()
    
    def save(self, path: str):
        np.save(path, self.grid)
    
    def load(self, path: str):
        self.grid = np.load(path)


if __name__ == "__main__":
    model = DielectricBreakdownModel(eta=5, dim=300)
    model.simulate()
    model.show()