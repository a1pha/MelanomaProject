import shutil
import pandas as pd
from tqdm import tqdm

df = pd.read_csv('data/ISIC_2019_Training_GroundTruth.csv')

for index, row in tqdm(df.iterrows()):
	src = 'data/ISIC_2019_Training_Input/' + row['image'] + '.jpg'
	if (row['MEL'] == 0):
		dst = './smaller_dataset/negative/'
		shutil.copy(src, dst)
	else:
		dst = './smaller_dataset/positive/'
		shutil.copy(src, dst)