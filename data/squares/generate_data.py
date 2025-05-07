### script to generate squares numerosity data 

import torch 
import numpy as np 
from random import randint 
from math import sqrt 
import pickle 

def check_overlap(image, h, w, posh, posw):
    # function: tests whether cur block over-laps with any pre-layed blocks
    # rejects if overlaps or within 1 pixel    
    [H, W] = image.size()
    minh = max(1, posh-1)
    maxh = min(H, posh+h)
    minw = max(1, posw-1)
    maxw = min(W, posw+w)

    block = image[minh-1:maxh-1, minw-1:maxw-1]
    #print(f"check_overlap: sum in image: {image.sum().item()}")
    if block.sum().item() == 0:
        return False
    else: 
        return True

D = torch.zeros((8*32*200, 28, 28)) 

index = 0

for N in range(1, 33): 
    for ia in range(8): 
        # choose cumulative area 
        cumA = 30 + 30 * randint(0, 8) 
        it = 1
        while it < 201: 
            count = 0
            tries = 0
            print(f"cumA: {str(cumA)}")
            run_cumA = cumA 
            while run_cumA > 0 and N > count: 
                # calculate the sides of the rectangle 
                A = run_cumA * 1.0 / (N - count) + 0.15 * np.random.standard_normal() 

                a = sqrt(A) 

                h = round(a + 0.3*np.random.standard_normal())
                w = round(a + 0.3*np.random.standard_normal())

                if h ==0:
                    h = 1
                if w ==0:
                    w = 1
                
                posh = randint(1, 28-h+1)
                posw = randint(1, 28-w+1)

                tries += 1 
                if tries > 300: 
                    it -= 1
                    break
                #print(f"h: {str(h)}, w: {str(w)}")
                #print(f"posh: {str(posh)}, posw: {str(posw)}")

                if not check_overlap(D[index, :, :], h, w, posh, posw): 
                    D[index, posh-1:posh+h-1, posw-1:posw+w-1] = 1
                    run_cumA = run_cumA - h * w
                    count += 1
                    print(f"run_cumA: {str(run_cumA)}")
                    print(f"h: {str(h)}, w: {str(w)}")
                    print(f"posh: {str(posh)}, posw: {str(posw)}")
                    print(f"N: {str(N)}")
                    print(f"count: {str(count)}")
                    print(f"it: {str(it)}")
                    print(f"tries: {str(tries)}")
            print(f"cumA: {str(cumA)}")
            #print(f"count: {str(count)}")
            print(f"processed {str(index)}th image")
            print(f"it: {str(it)}")
            index += 1
            it += 1
        
pickle.dump(D, 'squares_data.pkl') 
