# 2次元正規分布の描画
# sketch of 2-D normal distribution

# モジュールの読み込み
# import required python modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as st
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 正規分布の平均ベクトル（Mu1, Mu2）と共分散行列(Sigma1, Sigma2)の設定例
# set mean vectors（Mu1, Mu2）and covariance matrices (Sigma1, Sigma2)
Mu1 = np.array([5.0,0.0])
Mu2 = np.array([-5.0,0.0])
Sigma1 = np.array([[1.0,0.0],[0.0,1.0]])
Sigma2 = np.array([[1.0,0.0],[0.0,1.0]])

# 格子点の生成
# generate grid
x = y = np.arange(-10, 10, 0.1)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X,Y))

# 与えられた分布のposにおける確率密度値
# probability density value at pos for given distribution
Z1 = st.multivariate_normal.pdf(pos, mean=Mu1, cov=Sigma1)
Z2 = st.multivariate_normal.pdf(pos, mean=Mu2, cov=Sigma2)

Z = np.log(np.fmax(Z1, Z2))
Zdiff = Z1 - Z2

maxZ = np.max(Z)
minZ = np.min(Z)

# Zの最小値の設定
# set minimum value of Z
minZ = -20

# figureオブジェクトの生成
# generate figure object
fig = plt.figure(figsize=(7,5))

# subplotの書式設定
# format subplot
ax = fig.add_subplot(111, aspect='equal')

# 軸ラベルの設定
# set axis labels
ax.set_xlabel('x')
ax.set_ylabel('y')

# 等高線（等確率密度線）の描画
# sketch contours (equi-probability density contours)
levels = np.linspace(minZ, maxZ, 50)
cs = ax.contourf(X, Y, Z, levels=levels, cmap=cm.inferno, extend='both')

# 決定境界の描画
# sketch decision boundary
ax.contour(X, Y, Zdiff, levels=[-1.0e-300, 1.0e-300], colors='r', linestyles='-')

# 平均ベクトルの表示
# plot mean vectors
ax.scatter(Mu1[0], Mu1[1], s=40, c='red',  marker='o', alpha=0.5, linewidths=2)
ax.scatter(Mu2[0], Mu2[1], s=40, c='red',  marker='x', alpha=0.5, linewidths=2)

# カラー・バーの表示
# show color bar
# create the same size divider as the contour graph
divider = make_axes_locatable(ax)
# append 5% space to the right-end
cax = divider.append_axes('right', size='5%', pad=0.1)  
fig.colorbar(cs, cax=cax, format='%.1f')

# グリッドの表示
# show grid
ax.grid()

# グラフの表示
# show graph
plt.show()
