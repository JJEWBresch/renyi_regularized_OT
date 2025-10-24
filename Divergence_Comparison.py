# -*- coding: utf-8 -*-
"""
@author: Viktor Stein

This plots figure 2 from the preprint "Interpolating between Optimal Transport
and KL regularized Optimal Transport using Rényi Divergences" by J. Bresch
and V. Stein, available at https://arxiv.org/abs/2404.18834
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import cm  # for plotting color scheme


q = .25


def KL(p):
    return p*np.log(p/q) + (1 - p)*np.log((1 - p)/(1 - q))


def Renyi(p, a=2):
    if a == 1:
        return KL(p)
    else:
        return 1/(a - 1) * np.log((1/q)**(a - 1) * p**a + (1/(1 - q))**(a - 1) * (1 - p)**a)


def Tsallis(p, a=2):
    if a == 1:
        return KL(p)
    else:
        return 1/(a - 1) * ( (1 / q)**(a - 1) * p**a + (1 / (1 - q))**(a - 1) * (1 - p)**a - 1)


def alph(p, a=2):
    if a == 1:
        return KL(p)

    else:
        return 1/a * Tsallis(p, a)


x = np.linspace(0.01, 0.99, 1001)
# fig = plt.figure
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (12, 4))
# # fig.suptitle('Comparison of Renyi, Tsallis and alpha divergence')
# alphas = (.5, 1, 1.5, 2)
# markers = ('r', 'k:', 'b-', 'g-.')
# for alpha, m in zip(alphas, markers):
#     ax1.plot(x, Renyi(x, alpha), m, label = f'{alpha}')
#     ax2.plot(x, Tsallis(x, alpha), m, label = f'{alpha}')
#     ax3.plot(x, alph(x, alpha), m, label = f'{alpha}')
# ax1.legend(title=r'Value of $\alpha$', bbox_to_anchor=(2.5, -.1) , ncol = 4)
# plt.xlabel(r'$p$')
# # plt.ylabel(r'$R_{\alpha}$', loc = 'top')
# ax1.set_title('Renyi divergence')
# ax2.set_title('Tsallis divergence')
# ax3.set_title(r'$\alpha$ divergence')
# fig.savefig('Divergence_Comparison.png', bbox_inches='tight', dpi=300)
# plt.show()


L = 11*10
L2 = L + 3
c = [matplotlib.colors.rgb2hex(k) for k in cm.jet(np.array(range(L2))/L2)]
# the following is somehow wrong :()
# y = np.zeros(x.shape)
# for k in range(len(x)):
#     if x[k] < q:
#         yk = np.log(q/x[k])
#     else:
#         yk = np.log((1-q)/(1-x[k]))
#     y[k] = yk

# plt.plot(x, Renyi(x,1/4), c = c[0], label = f'{1/4}')
# plt.plot(x, Renyi(x,1/2), c = c[1], label = f'{1/2}')
# plt.plot(x, Renyi(x,3/4), c = c[2], label = f'{3/4}')
# plt.plot(x, Renyi(x,.9), c = c[2], label = f'{0.9}')
for t in range(110):
    plt.plot(x, Renyi(x, t/100), c=c[t+3], label=f'{t/100}')
# plt.plot(x, y, c = 'k', label=r'$\infty$')
# plt.legend(frameon=False)
plt.title(rf'Renyi divergence from $(p, 1 - p)$ to $({q}, {1-q})$')
plt.ylim([0, 1.5])
plt.savefig('Renyi_Comparison.pdf', dpi=300, bbox_inches='tight')

plt.show()

for t in range(110):
    plt.plot(x, Tsallis(x, t/100), c=c[t+3], label=f'{t/100}')
plt.title(rf'Tsalllis divergence from $(p, 1 - p)$ to $({q}, {1-q})$')
plt.ylim([0, 1.5])
plt.savefig('Tsallis_Comparison.pdf', dpi=300, bbox_inches='tight')
plt.show()
