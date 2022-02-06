import json


class DBTool:

    def __init__(self, db_path="db/db.json"):
        self.db_path = db_path
    

    def dump(self, location, latitude, longitude, img_path, seg_path):

        entry = {
            location : [
                {
                    f"({latitude}, {longitude})" : 
                    {"img_path" : img_path,
                     "seg_path" : seg_path, 
                    }
                }
            ]
        }

        with open(self.db_path, "r+") as jsonfile:

            db = json.load(jsonfile)
            if location in db:
                db[location].extend(entry[location])
            else:
                db.update(entry)
            jsonfile.seek(0)
            json.dump(db, jsonfile, indent=4)


if __name__ == "__main__":
    pass