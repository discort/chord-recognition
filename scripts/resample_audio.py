import os
import subprocess
import time


def resample(filename, out_dir, sample_rate=44100):
    out_path = filename.split('/')[-1]
    out_path = os.path.join(out_dir, out_path)
    subprocess.call(
        ['ffmpeg', '-i', filename, '-ar', str(sample_rate), out_path]
    )


def main():
    folders = (
        'data/beatles/mp3/A_Hard_Day_s_Night/',
        'data/beatles/mp3/With_The_Beatles',
        'data/beatles/mp3/Sgt_Pepper_s_Lonely_Hearts_Club_Band/',
        'data/beatles/mp3/Rubber_Soul/',
        'data/beatles/mp3/Revolver/',
        'data/beatles/mp3/Please_Please_Me/',
        'data/beatles/mp3/Help_/',
        'data/beatles/mp3/Beatles_For_Sale/',
    )
    #import pudb; pudb.set_trace()
    for folder in folders:
        for root, _, files in os.walk(folder):
            out_dir = os.path.join(root, 'temp')
            os.makedirs(out_dir)
            for f in files:
                filename = os.path.join(root, f)
                resample(filename, out_dir)
                time.sleep(2)


if __name__ == '__main__':
    main()
