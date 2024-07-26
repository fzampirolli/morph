import time

from morph import *

file = 'fig_Maze1.png'
#file = 'fig_Maze2.png'
# file = 'fig_Maze3.png'
img0 = mm.read(file)

if file == 'fig_Maze1.png':
    img0 = cv2.resize(img0,(35,35))
elif file == 'fig_Maze2.png':
    img0 = mm.ero(img0, mm.sebox())
    img0 = cv2.resize(img0, (700, 700))

imgRGB = mm.color(img0)
img = mm.gray(imgRGB)
f = (mm.threshold(img)/255).astype('uint16')

if file == 'fig_Maze2.png':
    f = mm.open(f,mm.secross())
    f1 = (mm.neg(f) / 255).astype('uint8')
    f2 = mm.label(f1)

plt.figure(figsize=(7, 7))
mm.show(f) # image (a)

m1 = np.zeros_like(f).astype('uint8')
m2 = np.zeros_like(f).astype('uint8')

if file == 'fig_Maze1.png':
    m1[1,17] = m2[-2,17] = 1
elif file == 'fig_Maze2.png':
    m1 = np.array(f2==2).astype('uint8')
    m2 = np.array(f2==3).astype('uint8')

start_time = time.time()  # Marcar o tempo de início

b = mm.gdist(f,m1) # image (b)

end_time = time.time()  # Marcar o tempo de término
# Calcular o tempo de processamento
processing_time = end_time - start_time
print(f"Processing time: {processing_time:.4f} seconds")

c = mm.gdist(f,m2) # image (c)

d = b + c
min = np.amin(d[d != np.amin(d)]) # minimum
print(f'minimum distance: {min}')
mm.show(imgRGB,d==min) # image (d)