import argparse

from loader import Loader
from segment import Segmenter


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--location", '-l', default="Luxembourg")
    parser.add_argument("--zoom", '-z', type=int, default=10)
    parser.add_argument("--img-size", type=int, default=2448)
    parser.add_argument("--img-dir", default="images")
    parser.add_argument("--seg-dir", default="segmented_images")
    parser.add_argument('--saved_model', '-m', default='saved_models/CP_epoch30.pth')
    parser.add_argument('--scale', '-s', type=float, default=0.8)
    parser.add_argument('--valid-idx', type=int, default=2)
    args = parser.parse_args()


    Loader().load(args.location, args.zoom, args.img_size, args.img_dir, args.seg_dir)
    
    segmenter = Segmenter(args.saved_model)
    segmenter.segment(args.location, args.img_dir, args.seg_dir, args.scale)

