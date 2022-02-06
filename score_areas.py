import json
from solarscore import SolarPower
import ast


class Scorer:

    def __init__(self, fname='PVOUT_local.tif'):
        self.fname = fname
        
        self.solarpower = SolarPower(fname)
        
        with open('db/db.json', 'r') as f:
            self.db = json.load(f)
    
    def score(self, location):

        scores = []
        for coord_str, entry in self.db[location].items():
            
            coords = ast.literal_eval(coord_str)
            frac = entry["frac"]
            solarscore = self.solarpower.get_solar_power(coords) * frac
            scores.append((coords, solarscore))

        scores.sort(key = lambda x:x[1], reverse=True)
        return scores

            

if __name__ == "__main__":
    scorer = Scorer()
    scores = scorer.score("Luxembourg")
    print(scores)