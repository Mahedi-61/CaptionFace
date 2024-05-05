import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
from sklearn.datasets import load_iris


"""
plt.figure()
iris = load_iris()
iris = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                    columns=iris['feature_names'] + ['target'])


chart = sns.displot(data=iris, x='sepal length (cm)', hue='target', kind='kde', fill=True, palette=sns.color_palette('bright')[:3], height=5, aspect=1.5)

## Changing title
new_title = 'This is a NEW title'
chart._legend.set_title(new_title)

# Replacing labels
new_labels = ['label 1', 'label 2', 'label 3']
for t, l in zip(chart._legend.texts, new_labels):
    t.set_text(l)

plt.show()
"""


import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

fig, axes = plt.subplots(1, 1, figsize=(10, 1))
#fig.subplots_adjust(hspace=4)


bins = [0, 0.5, 1, 2, 5]
nbin = len(bins) - 1
cmap4 = cm.get_cmap('rainbow', nbin)
norm4 = mcolors.BoundaryNorm(bins, nbin)
im4 = cm.ScalarMappable(norm=norm4, cmap=cmap4)

cbar4 = fig.colorbar(
    im4, cax=axes, orientation='horizontal',
    label='colorbar with BoundaryNorm'
)

plt.show()
#plt.savefig("fig.eps")
