
import matplotlib.pyplot as plt
import numpy as np
  
  

x_dataset_size=[4.36,8.62,9.87,21.0]
#------------------------MILESTONE 2--------------------------------------------------------------
#y_exec_time=[133.773997,742.177010,632.952988,3024.295092]#756.5,774.6,3807.3]
#y_kernel0_exec_time=[20.376001,80.875002,58.674999,264.667004]#62.694997]#,85.193001,87.337002]
#------------------------MILESTONE 3--------------------------------------------------------------
#y_cpu=[131.2,643.9,630.1,2788.6]
#y_k0=[19.0,88.5,60.5,277.8]
#y_k1=[13.3,50.9,51.7,576.2]
#------------------------MILESTONE 4--------------------------------------------------------------
#y_cpu=[117.16,661.77,781.83,2888.92]
#y_k0=[19.68,82.34,60.21,277.10]
#y_k1=[15.25,52.91,53.69,585.38]
#y_k2=[7.49,23.57,22.21,98.79]
#------------------------MILESTONE 5--------------------------------------------------------------
y_cpu=[141.1,606.8,622.3,2807.7]
y_k0=[19.2,87.8,64.6,284.6]
y_k1=[15.0,55.8,58.2,582.2]
y_k2=[7.1,22.8,21.6,97.7]
y_k3=[4.8,15.0,15.4,63.0]

plt.plot(x_dataset_size, y_cpu, color = 'black',
         linestyle = 'solid', marker = 'o',
         markerfacecolor = 'black', markersize = 3,label='CPU')
plt.plot(x_dataset_size, y_k0, color = 'blue',
         linestyle = 'solid', marker = 'o',
         markerfacecolor = 'black', markersize = 3,label='Kernel 0')
plt.plot(x_dataset_size, y_k1, color = 'green',
         linestyle = 'solid', marker = 'o',
         markerfacecolor = 'black', markersize = 3,label='Kernel 1')
plt.plot(x_dataset_size, y_k2, color = 'red',
         linestyle = 'solid', marker = 'o',
         markerfacecolor = 'black', markersize = 3,label='Kernel 2')
plt.plot(x_dataset_size, y_k3, color = 'goldenrod',
         linestyle = 'solid', marker = 'o',
         markerfacecolor = 'black', markersize = 3,label='Kernel 3')
plt.legend()
plt.xlabel("Dataset Size (MB)")
plt.ylabel("Execution Time (ms)")
plt.title("Execution Time versus Dataset Size")
plt.show()  
plt.figure()
