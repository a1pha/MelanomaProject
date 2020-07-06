import shutil
import pandas as pd
from tqdm import tqdm
import random

df = pd.read_csv('data/ISIC_2019_Training_GroundTruth.csv')

for index, row in tqdm(df.iterrows()):
	number = random.uniform(0, 1)
	positive = row['MEL'] == 1
	src = 'data/cropped/' + row['image'] + '.jpg'
	if number < 0.8:
		if positive:
			dst = './data/cropped_split/train/positive/'
		else:
			dst = './data/cropped_split/train/negative/'
	else:
		if positive:
			dst = './data/cropped_split/val/positive/'
		else:
			dst = './data/cropped_split/val/negative/'

	shutil.copy(src, dst)
