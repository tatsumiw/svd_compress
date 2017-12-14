import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl

def methods(sigma, u, v, K):         #svd方法的核心（奇异值，左奇异矩阵，右奇异矩阵，奇异值个数）
    uk = u[:, 0:K]                   #构建左奇异矩阵
    sigmak = np.diag(sigma[0:K])     #构建奇异值
    vk = v[0:K,]                     #构建右奇异矩阵
    a  = np.dot(np.dot(uk,sigmak),vk)#返回奇异值压缩后的矩阵
    a[a < 0] = 0                     #异常数值处理
    a[a > 255] = 255                 #异常数值处理
    return np.rint(a).astype('uint8')#整数化

if __name__ == "__main__":
    A = Image.open("jacky.jpg", 'r')
    output_path = r'.\Picture'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    a = np.array(A)   #把图像转化为矩阵
    K = 60            #奇异值的最大个数
    u_r, sigma_r, v_r = np.linalg.svd(a[:, :, 0]) #进行svd分解(返回一个元组)
    u_g, sigma_g, v_g = np.linalg.svd(a[:, :, 1]) #进行svd分解(返回一个元组)
    u_b, sigma_b, v_b = np.linalg.svd(a[:, :, 2]) #进行svd分解(返回一个元组)
    plt.figure(figsize=(10,10), facecolor='w')
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    i = 1
    while i <= 16:
        for k in range(1, K+1):
            print(k)
            R = methods(sigma_r, u_r, v_r, k)
            G = methods(sigma_g, u_g, v_g, k)
            B = methods(sigma_b, u_b, v_b, k)
            I = np.stack((R, G, B), 2)
            Image.fromarray(I).save('%s\\svd_%d.png' % (output_path, k))
            if k in [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:
                plt.subplot(4, 4, i)
                plt.imshow(I)
                plt.axis('off')
                plt.title(u'奇异值数目：%d' % k)
                i += 1
    plt.show()
