import json
import glob

states = glob.glob('data/mountains-moving/lake-line-0-north-low-20.0-0.05-default/states/1*.json')
states.sort()

for s in states:
    with open(s, 'r') as f:
        j = json.load(f)
        FoE = j['Drone1']['ue4']['FoE']
        print(f'{FoE['Y']:.02f})

#print(f'{0.05/(1920/821)*3:.04f}')
