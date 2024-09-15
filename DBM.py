import random
import numpy as np
import matplotlib.pyplot as plt

class Cell:
    def __init__(self, i, j, potential=0.):
        self.i = i
        self.j = j
        self.phi = potential

        self.phi_eta = None
        self.prob = None

    def get_coords(self):
        return self.i, self.j
    
    def get_neighbors(self):
        return [(self.i+1, self.j), (self.i-1, self.j), (self.i, self.j+1), (self.i, self.j-1), (self.i+1, self.j+1), (self.i+1, self.j-1), (self.i-1, self.j+1), (self.i-1, self.j-1)]
        

class DBM:
    # based on "Fast Simulation of Laplacian Growth" by Kim et al.
    # fuse breakdown instead of dielectric breakdown simulated

    def __init__(self, eta=1., dim=100, h=1):
        self.eta = eta
        # size of one grid block
        self.R1 = h/2

        # 0 = empty, -1 = candidate, 1 = filled
        self.grid = np.zeros((2*dim-1, 2*dim-1))

        self.pattern = []
        self.candidates = []
        self.hit_edge = False

        seed = Cell(dim, dim, 1)
        self.add_cell(seed)

    def grow_pattern(self):
        cell = self.choose_candidate()
        self.add_cell(cell)
        self.update_potentials(cell)

    def choose_candidate(self):
        min_phi = min(self.candidates, key=lambda c: c.phi).phi
        max_phi = max(self.candidates, key=lambda c: c.phi).phi

        for c in self.candidates:
            c.phi_eta = ((c.phi-min_phi)/(max_phi-min_phi))**self.eta

        phi_eta_sum = sum(c.phi_eta for c in self.candidates)
        for c in self.candidates:
            c.prob = c.phi_eta / phi_eta_sum

        idx = random.choices([*range(len(self.candidates))], weights=[c.prob for c in self.candidates])[0]
        cell = self.candidates.pop(idx)
        return cell
        
    def simulate(self, steps=None):
        if not steps:
            steps = 0
            while not self.hit_edge:
                self.grow_pattern()
                steps += 1
            return steps
        else:
            for _ in range(int(steps)):
                self.grow_pattern()

    def add_cell(self, cell):
        self.grid[cell.i, cell.j] = 1
        self.pattern.append(cell)

        possible_neighbors = cell.get_neighbors()
        neighbors = [p for p in possible_neighbors if 0 < p[0] < self.grid.shape[0] and 0 < p[1] < self.grid.shape[1]]
        if len(possible_neighbors) > len(neighbors):
            self.hit_edge = True
        neighbors = [p for p in neighbors if self.grid[p[0], p[1]] == 0]

        for n in neighbors:
            self.grid[n[0], n[1]] = -1        
        [self.candidates.append(Cell(p[0], p[1], self.calculate_spawn_potential(p[0], p[1]))) for p in neighbors]
        
    def calculate_spawn_potential(self, i, j):
        # equation 10
        return sum((1-self.R1/np.sqrt((i-c.i)**2+(j-c.j)**2)) for c in self.pattern)
    
    def update_potentials(self, cell):
        # equation 11
        for c in self.candidates:
            c.phi += 1 - self.R1 / np.sqrt((cell.i - c.i) ** 2 + (cell.j - c.j) ** 2)

    def show(self):
        img = self.grid
        img[np.where(img==-1)] = 0
        plt.imshow(img, cmap='afmhot')
        plt.axis('off')
        plt.show()
    
    def save_grid(self):
        np.save('grid.npy', self.grid)
