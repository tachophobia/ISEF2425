import os
import time
import numpy as np
from PIL import Image
from tqdm import tqdm

class GradientAnimation:
    """
    An animation class for generating GIFs from a directory of numpy matrices.
    The matrices are used to generate a gradient image where the color intensity is determined by the value in the matrix and the distance from the center.
    Intended specifically for animating the Dielectric Breakdown Model (DBM)
    """

    def __init__(self, directory_path, rate=1.1, inner_color=(173, 216, 230), outer_color=(0, 0, 139)):
        self.directory_path = directory_path
        self.inner_color = np.array(inner_color)
        self.outer_color = np.array(outer_color)
        self.rate = rate
        self.images = []
        self.end = None

    def _load_images(self):
        directory = sorted(os.listdir(self.directory_path), key=lambda n: int(n[:-4]))
        self.size = len(directory)
        f = 0
        last_f = f

        pbar = tqdm(total=self.size)
        while f < self.size:
            pbar.refresh()
            pbar.update(int(f-last_f))
            filename = directory[f]
            # mat = np.load(os.path.join(self.directory_path, filename))
            mat = np.genfromtxt(os.path.join(self.directory_path, filename), delimiter=',')
            mat = np.clip(mat * 255, 0, 255)

            height, width = mat.shape
            center = (width // 2, height // 2)
            max_distance = np.sqrt(center[0]**2 + center[1]**2)

            y, x = np.ogrid[:height, :width]
            distance_map = np.sqrt((x - center[0])**2 + (y - center[1])**2) / max_distance

            gradient_image = np.zeros((height, width, 3), dtype=np.uint8)
            for i in range(3):
                gradient_image[..., i] = (self.inner_color[i] - (self.inner_color[i] - self.outer_color[i]) * distance_map) * (mat / 255)
            self.images.append(Image.fromarray(gradient_image))

            last_f = f
            if f > self.size * 0.9:
                f += 10
                if self.end is None:
                    self.end = len(self.images)
            else:
                f = 1 + f * self.rate
                f = int(f)

    def _generate_reverse_sequence(self):
        reverse_images = []
        i = 0
        while i < self.end:
            reverse_images.append(self.images[self.end - 1 - i])
            i = 1 + i * self.rate
            i = max(int(i), 0)
        self.images.extend(reverse_images)

    def create_gif(self, output_path, duration=0, loop=0):
        print("Creating GIF...")
        start_time = time.perf_counter()
        self._load_images()
        self._generate_reverse_sequence()
        self.images[0].save(output_path, save_all=True, append_images=self.images[1:], duration=duration, loop=loop)
        end_time = time.perf_counter()
        print(f"GIF created in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    gif_generator = GradientAnimation('frames') # specify the frames directory
    gif_generator.create_gif('frames.gif')
