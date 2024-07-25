import time

from morph import *
#help(mm.gdist)
#print(cv2.__version__)

img0 = mm.read('fig_Maze.png')
imgRGB = mm.color(img0)
img = mm.gray(imgRGB)
f = (mm.threshold(img)/255).astype('uint16')
plt.figure(figsize=(7, 7))
mm.show(f) # image (a)

m1 = np.zeros_like(f).astype('uint8')
m2 = np.zeros_like(f).astype('uint8')
m1[1,17] = 1
m2[-2,17] = 1

start_time = time.time()  # Marcar o tempo de início
b = mm.gdist(f,m1) # image (b)
end_time = time.time()  # Marcar o tempo de término
# Calcular o tempo de processamento
processing_time = end_time - start_time
print(f"Tempo de processamento: {processing_time:.4f} segundos")

c = mm.gdist(f,m2) # image (c)

d = b + c
min = np.amin(d[d != np.amin(d)]) # minimum
print(min) # min == 17
mm.show(imgRGB,d==min) # image (d)