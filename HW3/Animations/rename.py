import os

rate = 5

for file in os.listdir():
    if file[:7] == "vortex_" and file[-4:] == ".jpg":
        number = int(file[7:11])
        # if number % rate == 0:
        #     os.rename(file, f"image_{number//rate:04d}.jpg")

        os.rename(file, f"image_{number//rate:04d}.jpg")

    # if file[:6] == "image_":
    #     number = int(file[6:10])
    #     os.rename(file, f"zalezak_{number*rate:04d}.jpg")
