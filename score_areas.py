import json
from solarscore import SolarPower


def append_solar_scores():
    solar = SolarPower()

    with open('db/db.json', 'r') as f:
        data = json.load(f)

    for coords in data['locations']:
        data['locations']['score'] = solar.get_solar_power(coords)

    return data
