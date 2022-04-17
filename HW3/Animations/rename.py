import os

rate = 5

for file in os.listdir():
    if file[:8] == "zalezak_" and file[-4:] == ".jpg":
        number = int(file[8:12])
        if number % rate == 0:
            os.rename(file, f"image_{number//rate:04d}.jpg")
    # if file[:6] == "image_":
    #     number = int(file[6:10])
    #     os.rename(file, f"zalezak_{number*rate:04d}.jpg")
