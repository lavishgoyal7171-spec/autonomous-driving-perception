"""Download public road test videos (Udacity self-driving-car course, MIT licence) into data/."""
import os
import sys
import urllib.request

VIDEOS = {
    "solidWhiteRight.mp4": "https://github.com/udacity/CarND-LaneLines-P1/raw/master/test_videos/solidWhiteRight.mp4",
    "solidYellowLeft.mp4": "https://github.com/udacity/CarND-LaneLines-P1/raw/master/test_videos/solidYellowLeft.mp4",
    "project_video.mp4": "https://github.com/udacity/CarND-Advanced-Lane-Lines/raw/master/project_video.mp4",
    "challenge_video.mp4": "https://github.com/udacity/CarND-Advanced-Lane-Lines/raw/master/challenge_video.mp4",
}


def main(names):
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(root, exist_ok=True)
    for name, url in VIDEOS.items():
        if names and name not in names:
            continue
        dst = os.path.join(root, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"[skip] {name} already exists")
            continue
        print(f"[download] {name} ...", flush=True)
        urllib.request.urlretrieve(url, dst)
        print(f"           saved {os.path.getsize(dst) / 1e6:.1f} MB -> {dst}")


if __name__ == "__main__":
    main(set(sys.argv[1:]))
