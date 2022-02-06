import json
from solarscore import SolarPower
import ast


def append_solar_scores():

    solarpower = SolarPower(fname='PVOUT_local.tif')

    with open('db/db.json', 'r') as f:
        db = json.load(f)


    for location, entries in db.items():
        for entry in entries:
            coord_str = list(entry.keys())[0]
            coords = ast.literal_eval(coord_str)
            solarscore = solarpower.get_solar_power(coords)
            

if __name__ == "__main__":
    append_solar_scores()