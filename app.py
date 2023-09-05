import pygame , sys
import numpy as np
from pygame.locals import *
from keras.models import load_model
import cv2
WHITE= (255,255,255)
MODEL =load_model("/home/lee/Downloads/PLant_Disease/practice/TRAIN/my_model.h5")
pygame.init()
BOUND=5
WINDOWSIZEX=640
WINDOWSIZEY=480
IMAGESAVE=False
DISPLAYSURFACE = pygame.display.set_mode((640,480))
WHITE_INIT = DISPLAYSURFACE.map_rgb(WHITE)

pygame.display.set_caption("DEMO")
PREDICT=True
imwriting= False
num_xcord=[]
num_ycord=[]
img_cnt=1

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type== MOUSEMOTION and imwriting:
            xcord, ycord= event .pos
            pygame.draw.circle(DISPLAYSURFACE, WHITE,(xcord,ycord),4,0)
            num_xcord.append(xcord)
            num_ycord.append(ycord)
        if event.type== MOUSEBUTTONDOWN:
            imwriting=True    
        if event.type== MOUSEBUTTONUP:
            imwriting=False
            num_xcord=sorted(num_xcord)
            num_ycord=sorted(num_ycord) 

            rect_min_x,rect_max_x  = max(num_xcord[0]-BOUND, 0), min(WINDOWSIZEX, num_xcord[-1]+BOUND) 
            rect_min_y,rect_max_y  = max(num_ycord[0]-BOUND, 0), min(WINDOWSIZEY, num_ycord[-1]+BOUND) 

            num_xcord=[]
            num_ycord=[]
            img_Arr= np.array(pygame.PixelArray(DISPLAYSURFACE))
            if IMAGESAVE: 
                cv2.imwrite("image.png")
                img_cnt+=1
            if PREDICT: 
                image=cv2.resize(img_Arr,(28,28))
                image=np.pad(image,(10,10),'constant',constant_values=0)
                image= cv2.resize(image,(28,28))/WHITE_INIT
                labe=str(LABELS[np.argmax(MODEL.predict(image.reshape(1,28,28,-1)))]).title()
                pygame.draw.rect (DISPLAYSURFACE, RED, (rect_min_x, rect_min_y, rect_max_x-rect_min_y, rect_max_y-rect_min_y), 3)
                
            if event.type == KEYDOWN:
                if event.unicode == 'N': 
                    DISPLAYSURFACE.fill(BLACK)
                
                
        pygame.display.update()
