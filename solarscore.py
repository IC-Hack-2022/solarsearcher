import rasterio
import numpy as np


class SolarPower:
    def __init__(self):
        with rasterio.open('data/PVOUT.tif') as dataset:
            self.data_numpy = dataset.read()

    def _coords_to_index(self, coords):
        lat, long = coords
        assert(-50 < lat < 60 and -180 < long < 180)
        with rasterio.open('data/PVOUT.tif') as dataset:
            return 0, *dataset.index(long, lat)

    def get_solar_power(self, coords):
        index = self._coords_to_index(coords)
        print(index)
        power = self.data_numpy[index]
        if np.isnan(power):
            raise ValueError("no power at coord")
        else:
            return power


if __name__ == '__main__':
    solar = SolarPower()
    power = solar.get_solar_power((51.51, -0.19))
    print(power)
