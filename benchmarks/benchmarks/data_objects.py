import yt

class TimeSuite:

    def setup(self):
        self.ds = yt.load("IsolatedGalaxy/galaxy0030/galaxy0030")
        # warm up disk cache
        self.time_projection()

    def time_slice(self):
        sl = self.ds.slice(2, 0.5)
        sl["density"]

    def time_projection(self):
        self.ds.proj("density", "z")

    def time_sphere(self):
        sp = self.ds.sphere([0.5, 0.5, 0.5], (2, 'kpc'))
        sp["density"]
