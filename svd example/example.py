from imageio import imread, imwrite
from numpy.linalg import svd
from numpy import outer

image = imread('hbsZF.png')
print(image.shape)


u, s, vh = svd(image)
print(u.shape, s.shape, vh.shape)

im = outer(u[0],vh[0])
imwrite('out.png',im)