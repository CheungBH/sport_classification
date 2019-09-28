import os
import random

main_path = r'../data\golf_ske\train\standing'
select_num = 800

pic_path = os.listdir(main_path)
numList = random.sample(range(0, len(pic_path)), len(pic_path) - select_num)
numList.sort(reverse=True)
for del_num in numList:
    os.remove(os.path.join(main_path, pic_path[del_num]))

