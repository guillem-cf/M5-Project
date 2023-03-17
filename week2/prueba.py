import pandas as pd

df = pd.read_csv('kitti-mots_annotations.csv')

print((df[(df['video_id']==0) & (df['frame_id']=='000044.png')]))