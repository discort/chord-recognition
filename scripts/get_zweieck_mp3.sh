# mkdir Zweieck && cd Zweieck

# Download Zweieck playlist
# youtube-dl -i -f mp4 --yes-playlist 'https://www.youtube.com/watch?v=HsVOEMEBBuM&list=PLeyqLueQDXJak8I9sf-v-R1X4sgGDkByM'

# Convert .mp4 files to .mp3
for i in *.mp4;
  do name=`echo "$i" | cut -d'.' -f1`
  echo "$name"
  ffmpeg -i "$i" -acodec "libmp3lame" "${name}.mp3"
done

# Remove all .mp4 files
# find . -name "*.mp4" -type f -delete