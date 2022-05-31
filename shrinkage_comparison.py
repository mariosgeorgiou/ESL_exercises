

import matplotlib.pyplot as plt
import matplotlib.font_manager
import numpy as np
import seaborn as sns

sns.set(font='Franklin Gothic Book',
        rc={
 'axes.axisbelow': False,
 'axes.edgecolor': 'lightgrey',
 'axes.facecolor': 'None',
 'axes.grid': False,
 'axes.labelcolor': 'dimgrey',
 'axes.spines.right': False,
 'axes.spines.top': False,
 'figure.facecolor': 'white',
 'lines.solid_capstyle': 'round',
 'patch.edgecolor': 'w',
 'patch.force_edgecolor': True,
 'text.color': 'dimgrey',
 'xtick.bottom': False,
 'xtick.color': 'dimgrey',
 'xtick.direction': 'out',
 'xtick.top': False,
 'ytick.color': 'dimgrey',
 'ytick.direction': 'out',
 'ytick.left': False,
 'ytick.right': False})
 
sns.set_context("notebook", rc={"font.size":16,
                                "axes.titlesize":20,
                                "axes.labelsize":18})

# 100 linearly spaced numbers
X = np.linspace(-2,2,100)

def f(x):
    return x

def g(x):
    return x if abs(x)>1 else 0

def h(x):
    return x/4

def i(x):
    return np.sign(x)*(abs(x)-1 if abs(x)-1 > 0 else 0)

y1 = f(X)
y2 = list(map(g, X))
y3 = h(X)
y4 = list(map(i, X))



plt.plot(X, y1, linestyle='dashed', color='#2CBDFE', label='Least Squares')
plt.plot(X, y2, color='#47DBCD', label='Subset Selection')
plt.plot(X, y3, color='#F3A0F2', label='Ridge')
plt.plot(X, y4, color='#9D2EC5', label='Lasso')
  
# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Least Squares Coefficient")
plt.ylabel("Transformed coefficient")
plt.title("Shrinkage Methods Comparison")
  
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()
  
# To load the display window
plt.show()

# # the function, which is y = x^2 here
# y = x

# # setting the axes at the centre
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.spines['left'].set_position('center')
# ax.spines['bottom'].set_position('zero')
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')

# # plot the function
# plt.plot(x,y, 'r')

# # show the plot
# plt.show()

