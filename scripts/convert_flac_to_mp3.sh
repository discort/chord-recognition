#!/bin/bash
for f in *.flac; do
    ffmpeg -i "$f" -ab 320k -map_metadata 0 -ar 44100 -id3v2_version 3 "${f%.flac}.mp3"
done