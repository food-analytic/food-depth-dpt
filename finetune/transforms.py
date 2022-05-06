import random


class CutDepth:
    def __init__(self, p=0.5, p_param=0.5):
        self.p = p
        self.p_param = p_param

    def __call__(self, image, depth):
        if random.random() < self.p:
            H, W, _ = image.shape

            a = random.random()
            b = random.random()
            c = random.random()
            d = random.random()

            l = int(a * W)
            u = int(b * H)
            w = int(min((W - a * W) * c * self.p_param, 1))
            h = int(min((H - b * H) * d * self.p_param, 1))

            image[u : u + h, l : l + w, 0] = depth[u : u + h, l : l + w]
            image[u : u + h, l : l + w, 1] = depth[u : u + h, l : l + w]
            image[u : u + h, l : l + w, 2] = depth[u : u + h, l : l + w]

        return image
