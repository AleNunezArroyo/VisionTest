import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

# Source files https://github.com/AlejandroNunezArroyo/VisionTest
# Model in h5 https://github.com/AlejandroNunezArroyo/VisionTest/blob/main/FaceDataset/FaceDataset.h5
# Images https://github.com/AlejandroNunezArroyo/VisionTest/tree/main/FaceDataset/images
model=load_model("/home/ale/Documents/GitHub/FaceDataset/FaceDataset/FaceDataset.h5")

img = cv2.imread('/home/ale/Documents/GitHub/FaceDataset/FaceDataset/images/SINMASK/0.jpg')

im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
rerect_sized=cv2.resize(im,(300,300))
normalized=rerect_sized/255.0
reshaped=np.reshape(normalized,(1,300,300,3))
reshaped = np.vstack([reshaped])
result=model.predict(reshaped)

if result > 0.5:
  print(" Sin mascara ")
  cv2.imshow('Imagen sin mascara',img)
else: 
  print(" Con mascara ")
  cv2.imshow('Imagen con mascara',img)
print(result)
cv2.waitKey(0)
cv2.destroyAllWindows()