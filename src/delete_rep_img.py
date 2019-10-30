import os

folder = '../data/src/image/Yoga/tree'
step = 5

imgs = os.listdir(folder)
for idx, img in enumerate(imgs):
    if idx % step != 0:
        os.remove(os.path.join(folder, img))
    else:
        pass

