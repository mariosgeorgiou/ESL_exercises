from imageio import imread, imwrite
from numpy.linalg import svd
from numpy import outer

image = imread('hbsZF.png')

u, s, vh = svd(image)

result = sum([s[i]*outer(u[:, i], vh[i]) for i in range(0, 1)])

imwrite('out.png', result)
