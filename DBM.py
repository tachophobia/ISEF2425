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
        

class DielectricBreakdownModel:
    # based on "Fast Simulation of Laplacian Growth" by Kim et al.
    # fuse breakdown instead of dielectric breakdown simulated for effectively same result

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
        phi_values = np.array([c.phi for c in self.candidates])
        min_phi = np.min(phi_values)
        max_phi = np.max(phi_values)
        
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
        
    def animate(self, steps: int):
        frames = []
        for _ in range(steps):
            self.grow_pattern()
            frames.append(self.grid.copy())
        
        return steps, frames
            
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
        
        for p in neighbors:
            self.candidates.append(Cell(p[0], p[1], self.calculate_spawn_potential(p[0], p[1])))

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
        pattern_coords = np.array([(c.i, c.j) for c in self.pattern])
        pattern_distances = np.sqrt((i - pattern_coords[:, 0]) ** 2 + (j - pattern_coords[:, 1]) ** 2)
        neighbor_potential = np.sum(1 - self.R1 / pattern_distances)

        environment_potential = np.sum([c.phi / np.sqrt((i - c.i) ** 2 + (j - c.j) ** 2) for c in self.environment])
        return neighbor_potential + environment_potential

    
    def update_potentials(self, cell):
        # equation 11
        candidate_coords = np.array([(c.i, c.j) for c in self.candidates])
        distances = np.sqrt((candidate_coords[:, 0] - cell.i) ** 2 + (candidate_coords[:, 1] - cell.j) ** 2)
        delta_potentials = 1 - self.R1 / distances
        for c, delta in zip(self.candidates, delta_potentials):
            c.phi += delta


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
    
    def save(self, path: str):
        np.save(path, self.grid)
    
    def load(self, path: str):
        self.grid = np.load(path)
