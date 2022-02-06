from email.mime import image
import json
import requests
import math as m
import pprint
import argparse
import os

class Loader:

    def __init__(self):

        with open('utils/bounding_boxes.json', 'r') as f:
            self.country_bounding_boxes = json.load(f)


    def load(self, location, zoom, img_size, outdir):

        image_coords = self.get_image_coords(location, zoom, img_size)

        print(f"Number of images to download: {len(image_coords)}.")

        outpath = os.path.join(outdir, location)
        if not os.path.exists(outpath):
            os.makedirs(outpath)
            self.load_country_images(image_coords, location, outpath)
        else:
            print(f"Images for country {location} already downloaded.")


    def metres_per_latitude(self, lat):
        lat_rad = lat * m.pi / 180
        return 111132.92 - (559.82 * m.cos(2 * lat_rad)) \
            + (1.175 * m.cos(4 * lat_rad)) - (0.0023 * m.cos(6 * lat_rad))


    def metres_per_longitude(self, lat):
        lat_rad = lat * m.pi / 180
        return (111412.84 * m.cos(lat_rad)) - (93.5 * m.cos(3 * lat_rad)) \
            + (0.118 * m.cos(5 * lat_rad))


    def bounding_box_size(self, lat_range, long_range):
        [min_lat, max_lat] = lat_range
        [min_long, max_long] = long_range

        lat_difference = (max_lat - min_lat)
        long_difference = (max_long - min_long)

        min_metre_per_long = self.metres_per_longitude(max_lat)
        max_metre_per_long = self.metres_per_longitude(min_lat)

        min_metre_per_lat = self.metres_per_latitude(max_lat)
        max_metre_per_lat = self.metres_per_latitude(min_lat)

        min_lat_width = long_difference * min_metre_per_long
        max_lat_width = long_difference * max_metre_per_long

        min_lat_height = lat_difference * min_metre_per_lat
        max_lat_height = lat_difference * max_metre_per_lat

        height = (min_lat_height + max_lat_height) / 2
        width = (min_lat_width + max_lat_width) / 2

        return height, width


    def distance_across_image(self, zoom, lat, image_size):
        metres_per_pixel = 156543.03392 * m.cos(lat * m.pi / 180) / (2 ** zoom)
        return metres_per_pixel * image_size


    def generate_coords(self, lat, longs, cols):
        row_coords = list()

        for i in range(cols):
            long = (0.5 + i) * (longs[1] - longs[0]) / cols + longs[0]
            row_coords.append((round(lat, 2), round(long, 2)))
        
        return row_coords


    def get_image_coords(self, country_name, zoom, image_size):
        if country_name not in self.country_bounding_boxes:
            raise ValueError(f"Country {country_name} doesn't exist!")

        country_coords = self.country_bounding_boxes[country_name]

        lats = country_coords["lat"]
        longs = country_coords["long"]

        mean_lat = sum(lats) / 2

        country_height, country_width = self.bounding_box_size(lats, longs)
        image_distance = self.distance_across_image(zoom, mean_lat, image_size)

        rows = m.ceil(country_height / image_distance)
        cols = m.ceil(country_width / image_distance)

        coords = list()

        for i in range(rows):
            lat = (0.5 + i) * (lats[1] - lats[0]) / rows + lats[0]
            coords.extend(self.generate_coords(lat, longs, cols))
        
        return coords


    def load_country_images(self, image_coords, country, directory):
        for coord in image_coords:
            lat, long = coord
            res = requests.get(
                "https://maps.googleapis.com/maps/api/staticmap",
                params={
                    "center": f"{lat},{long}",
                    "size": "2448x2448",
                    "scale": "2",
                    "zoom": "11",
                    "maptype": "satellite",
                    "key": "AIzaSyBDn2wVZ3iyViyiTrlKvFvOCCgmffuKc7w",
                    "format": "jpg"
                })

            if not res.ok:
                pprint.pprint(res.__dict__)
                raise Exception("Request went wrong!")
            
            with open(f"{directory}/{country}_{lat}_{long}.jpg", 'wb') as f:
                f.write(res.content)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--location", '-l', default="Luxembourg")
    parser.add_argument("--zoom", '-z', type=int, default=10)
    parser.add_argument("--img-size", type=int, default=2448)
    parser.add_argument("--outdir", default="images")
    args = parser.parse_args()

    Loader().load(args.location, args.zoom, args.img_size, args.outdir)
