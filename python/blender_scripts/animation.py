import bpy
import numpy as np

C = bpy.context
D = bpy.data

filepath = "/Users/philipp/Documents/GitHub/stage_cmap/python/"

def to_tuples(pos):
    n = len(pos)
    
    tuples = []
    
    for i in range(n):
        new_tuple = (pos[i, 0], pos[i, 1], pos[i, 2])
        tuples.append(new_tuple)
        
        
    return tuples


sphere1 = D.objects['sphere1']
sphere2 = D.objects['sphere2']
sphere3 = D.objects['sphere3']
sphere4 = D.objects['sphere4']
centersphere = D.objects['centersphere']

locations1 = np.genfromtxt(filepath + 'loc1.csv', delimiter=',')
locations2 = np.genfromtxt(filepath + 'loc2.csv', delimiter=',')
locations3 = np.genfromtxt(filepath + 'loc3.csv', delimiter=',')
locations4 = np.genfromtxt(filepath + 'loc4.csv', delimiter=',')
locationsc = np.genfromtxt(filepath + 'locc.csv', delimiter=',')

locations1 = to_tuples(locations1)
locations2 = to_tuples(locations2)
locations3 = to_tuples(locations3)
locations4 = to_tuples(locations4)
locationsc = to_tuples(locationsc)

spheres = [sphere1, sphere2, sphere3, sphere4, centersphere]
locations = [locations1, locations2, locations3, locations4, locationsc]

frame_num = 1

for n in range(len(locations1)):
    C.scene.frame_set(frame_num)
    newlocs = [locations1[n], locations2[n], locations3[n], locations4[n], locationsc[n]]
    
    for sphere, newloc in zip(spheres,newlocs):
        print(sphere)
        sphere.location = newloc
        sphere.keyframe_insert(data_path="location", index = -1)
        
    frame_num += 1