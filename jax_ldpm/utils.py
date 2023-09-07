import os
import json


def make_video(data_dir):
    # The command -pix_fmt yuv420p is to ensure preview of video on Mac OS is
    # enabled
    # https://apple.stackexchange.com/questions/166553/why-wont-video-from-ffmpeg-show-in-quicktime-imovie-or-quick-preview
    # The command -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" is to solve the following
    # "not-divisible-by-2" problem
    # https://stackoverflow.com/questions/20847674/ffmpeg-libx264-height-not-divisible-by-2
    # -y means always overwrite

    # TODO
    os.system(
        f'ffmpeg -y -framerate 10 -i {data_dir}/png/tmp/u.%04d.png -pix_fmt yuv420p -vf \
               "crop=trunc(iw/2)*2:trunc(ih/2)*2" {data_dir}/mp4/test.mp4') # noqa


def json_parse(json_filepath):
    with open(json_filepath) as f:
        args = json.load(f)
    json_formatted_str = json.dumps(args, indent=4)
    print(json_formatted_str)
    return args