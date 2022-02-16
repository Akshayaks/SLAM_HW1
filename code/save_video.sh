ffmpeg -r 30 -f image2 -i %04d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p test.mp4
