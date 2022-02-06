import rasterio
import numpy as np


class SolarPower:
    def __init__(self, fname='PVOUT_local.tif'):
        self.fname = fname
        with rasterio.open(self.fname) as dataset:
            self.data_numpy = dataset.read()

    def _coords_to_index(self, coords):
        lat, long = coords
        assert(-50 < lat < 60 and -180 < long < 180)
        with rasterio.open(self.fname) as dataset:
            return 0, *dataset.index(long, lat)

    def get_solar_power(self, coords):
        index = self._coords_to_index(coords)
        power = self.data_numpy[index]
        if np.isnan(power):
            raise ValueError(f"No power at coords {coords}.")
        else:
            return power


if __name__ == '__main__':
    pass