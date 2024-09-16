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

    def __init__(self, eta=1., dim=100, h=1, seed=None):
        self.eta = eta
        # size of one grid block
        self.R1 = h/2

        # 0 = empty, 1 = filled
        self.dim = dim
        self.grid = np.zeros((2*dim+1, 2*dim+1))

        self.pattern = []
        self.environment = []
        self.candidates = []
        self.hit_edge = False

        if seed:
            seed = Cell(seed[0], seed[1], 1)
        else:
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
        
    def simulate(self, steps=None, animate=False):
        frames = []
        if not steps:
            steps = 0
            while not self.hit_edge:
                self.grow_pattern()
                steps += 1
                if animate:
                    frames.append(self.grid.copy())
        else:
            for _ in range(int(steps)):
                self.grow_pattern()
                if animate:
                    frames.append(self.grid.copy())
        if frames:
            return steps, frames
        else:
            return steps
            

    def add_cell(self, cell):
        self.grid[cell.i, cell.j] = 1
        self.pattern.append(cell)

        possible_neighbors = cell.get_neighbors()
        neighbors = [p for p in possible_neighbors if 0 < p[0] < self.grid.shape[0] and 0 < p[1] < self.grid.shape[1]]
        if len(possible_neighbors) > len(neighbors):
            self.hit_edge = True
        # only allow for neighbors in neutral space
        neighbors = [p for p in neighbors if self.grid[p[0], p[1]] == 0]

        for n in neighbors:
            self.grid[n[0], n[1]] = -1        
        [self.candidates.append(Cell(p[0], p[1], self.calculate_spawn_potential(p[0], p[1]))) for p in neighbors]

    def add_line_of_charge(self, p1, p2, phi=-1):
        # create a line starting from p1 to p2
        x0, y0, x1, y1 = p1[0], p1[1], p2[0], p2[1]
        dx = abs(x1-x0)
        sx = [-1, 1][x0 < x1]
        dy = -abs(y1-y0)
        sy = [-1, 1][y0 < y1]
        error = dx + dy

        while True:
            if not (0 <= x0 < self.grid.shape[0] and 0 <= y0 < self.grid.shape[1]):
                break
            cell = Cell(x0, y0, phi)
            self.grid[x0, y0] = phi

            self.environment.append(cell)

            if (x0 == x1 and y0 == y1):
                break
            e2 = 2 * error
            if e2 > dy:
                error += dy
                x0 += sx
            if e2 < dx:
                error += dx
                y0 += sy
            x0 = int(x0)
            y0 = int(y0)

    def add_circle_of_charge(self, center, radius, phi=-1):
        # create a circle centered at center with radius radius
        x0, y0 = center
        theta = 0
        visited = set()
        while theta < 2*np.pi:
            x = int(x0 + radius * np.cos(theta))
            y = int(y0 + radius * np.sin(theta))
            if (x, y) not in visited and 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1]:
                cell = Cell(x, y, phi)
                self.grid[x, y] = phi
                self.environment.append(cell)
            theta += 0.01
            visited.add((x, y))
    
    def add_rectangle_of_charge(self, p1, p2, phi=-1):
        x0, y0, x1, y1 = int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])
        for x in range(x0, x1):
            for y in range(y0, y1):
                if 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1]:
                    cell = Cell(x, y, phi)
                    self.grid[x, y] = phi
                    self.environment.append(cell)
        
    def calculate_spawn_potential(self, i, j):
        # equation 10
        neighbor_potential = sum((1-self.R1/np.sqrt((i-c.i)**2+(j-c.j)**2)) for c in self.pattern)
        environment_potential = sum(c.phi/np.sqrt((i-c.i)**2+(j-c.j)**2) for c in self.environment)
        return neighbor_potential + environment_potential
    
    def update_potentials(self, cell):
        # equation 11
        for c in self.candidates:
            c.phi += 1 - self.R1 / np.sqrt((cell.i - c.i) ** 2 + (cell.j - c.j) ** 2)

    def show(self, show_environment=False):
        img = self.grid
        if not show_environment:
            img[np.where(img!=1)] = 0
            plt.imshow(img, cmap='afmhot')
        else:
            img[np.where(img > 1)] = 2
            img[np.where(img < -1)] = -2
            plt.imshow(img)
        plt.axis('off')
        plt.show()
    
    def save_grid(self):
        np.save('grid.npy', self.grid)
    
    def load_grid(self, path):
        self.grid = np.load(path)
