### script to generate squares numerosity data 

import torch 
import numpy as np 
from random import randint 
from math import sqrt 
import pickle 
import pdb 
import matplotlib.pyplot as plt

def tile_images(images, rows, cols):
    fig, axes = plt.subplots(rows, cols)
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i])
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')
    plt.show()
def check_overlap(image, h, w, posh, posw):
    # function: tests whether cur block over-laps with any pre-layed blocks
    # rejects if overlaps or within 1 pixel    
    [H, W] = image.size()
    #minh = max(1, posh-1)
    #maxh = min(H, posh+h)
    #minw = max(1, posw-1)
    #maxw = min(W, posw+w)

    block = image[posh-1:posh+h-1, posw-1:posw+w-1]
    #print(f"check_overlap: sum in image: {image.sum().item()}")
    if block.sum().item() == 0:
        return False
    else:
        print(f"image.sum().item():{str(image.sum().item())}") 
        return True
        

D = torch.zeros((8*32*200, 28, 28)) 
L = torch.zeros((8*32*200, 1)) 

index = 0

for N in range(1, 33): 
    for ia in range(8): 
        # choose cumulative area 
        cumA = 24 + 24 * randint(0, 8) 
        print(f"cumA: {str(cumA)}")
        print(f"N:{str(N)}")
        for s in range(200): 
            img = torch.zeros((28, 28)) 
            running_cumA = cumA 
            count = 0 
            while running_cumA > 0 and N > count: 
                A = running_cumA * 1.0 / (N - count) + 0.15 * np.random.standard_normal() 
                a = sqrt(max(1e-7, A))
                h = round(a + 0.15*np.random.standard_normal())
                h = 1 if h <= 0 else h
                w = round(a + 0.15*np.random.standard_normal())
                w = 1 if w <= 0 else w
                posh = randint(1, 28-h+1)
                posw = randint(1, 28-w+1)
                overlap = check_overlap(img, h, w, posh, posw)
                if not overlap: 
                    img[posh-1: posh+h-1, posw-1: posw+w-1] = 1
                    running_cumA = running_cumA - h * w
                    count += 1
                    print(f"A:{str(A)}")
                    print(f"a:{str(a)}")
                    print(f"h:{str(h)}, w:{str(w)}")
                    print(f"posh:{str(posh)}, posw:{str(posw)}")
                    print(f"overlap:{str(overlap)}")
                    print(f'img_sum:{str(img.sum().item())}')
                    print(f"count: {str(count)}")
                    print(f"running_cumA: {str(running_cumA)}")
                    
            D[index, :, :] = img
            L[index] = count
            index += 1
            print(f"processed {str(index)}th image")

data = {'D': D, 'L': L}

with open('squares_data.pkl', 'wb') as file:
    pickle.dump(data, file)

indices = torch.arange(0, len(D), 800)
dd = torch.index_select(D, 0, indices)
tile_images(dd, 8, 8)

pdb.set_trace()

