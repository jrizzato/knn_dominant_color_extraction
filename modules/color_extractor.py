import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
import numpy as np

img =cv2.imread('./data/image.jpg')
print(img.shape) #(4000, 6000, 3)

plt.imshow(img)
plt.show()

# la imagen se ve mal
# el problema es que el imread la carga conel formato BGR (en lugar de RGB)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

X = img.reshape((-1, 3)) # el -1 calcula automatamente la dimension restante
print(X.shape) # son 24M pixel, es mucho. Le voy a cambiar el tamaño a la imagen

img = cv2.resize(img, (400, 600))
X = img.reshape((-1, 3)) 
print(X.shape) 

k = 5
model = KMeans(n_clusters=k)
model.fit(X)

centroids = model.cluster_centers_

print(centroids) # matriz de 3 columnas (R, G, B) y 4 filas (porque elegimos k = 4, 4 colores dominantes)
#                   R           G           B
#  color 1: [ 72.74659474 126.08128793 154.11340189]
#  color 2: [  6.54190206  19.84364807  29.13968064]
#  color 3: [ 42.45817908  92.40691473 119.6461536 ]
#  color 4: [117.77513745 167.04828276 191.74207783]

colors = np.array(centroids, dtype="uint8") # redondeo a entero sin signo de 8 bits
#        R   G   B
#  C1: [ 72 126 154]
#  C2: [  6  19  29]
#  C3: [ 42  92 119]
#  C4: [117 167 191]
print(colors)

mat = np.zeros((100, 100, 3), dtype='uint8')
mat[:, :, 0] = 255 # cuadrado rojo
# mat[:, :, 1] = 255 # cuadrado verde
# mat[:, :, 2] = 255 # cuadrado azul

plt.imshow(mat)
plt.show()

for i, color in enumerate(colors):
    plt.subplot(1, k, i+1)
    mat = np.zeros((100, 100, 3), dtype='uint8')
    mat[:,:,:] = color
    plt.imshow(mat)
    plt.title(f'RGB({color[0]}, {color[1]}, {color[2]})')
    plt.axis('off')

plt.show()