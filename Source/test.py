# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 09:20:37 2019

@author: mai2125
"""
import matplotlib.pyplot as plt
import lynch_presets as ps

x = [0, 2, 4, 6]
y = [1, 3, 4, 8]


plt.plot(x,y)


plt.xlabel('x values')
plt.ylabel('y values')
plt.title('plotted x and y values')
plt.legend(['line 1'])


# save the figure
plt.savefig(ps.src/'plot.png', dpi=300, bbox_inches='tight')


plt.show()

plt.close()