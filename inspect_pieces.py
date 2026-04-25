import torch, glob

pieces = {}
for p in sorted(glob.glob('/var/home/jeremy/Development/jane_street_dropped/pieces/*.pth')):
    idx = int(p.split('_')[-1].split('.')[0])
    sd = torch.load(p, map_location='cpu', weights_only=True)
    pieces[idx] = sd

shapes = {}
for idx, sd in pieces.items():
    w = sd['weight']
    shapes[w.shape] = shapes.get(w.shape, 0) + 1

print('Total pieces:', len(pieces))
print('Shapes:', shapes)

import pandas as pd
df = pd.read_csv('/var/home/jeremy/Development/jane_street_dropped/historical_data.csv')
print('Data shape:', df.shape)
print('Columns:', df.columns.tolist())
