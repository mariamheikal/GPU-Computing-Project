
import matplotlib.pyplot as plt
import numpy as np
  
  

x_dataset_size=[4.36,8.62,9.87,21.0]
y_cpu_exec_time=[133.773997,742.177010,632.952988,3024.295092]
y_kernel0_exec_time=[20.376001,80.875002,58.674999,264.667004]
plt.plot(x_dataset_size, y_cpu_exec_time, color = 'green',
         linestyle = 'solid', marker = 'o',
         markerfacecolor = 'black', markersize = 5)
plt.plot(x_dataset_size, y_kernel0_exec_time, color = 'blue',
         linestyle = 'solid', marker = 'o',
         markerfacecolor = 'black', markersize = 5)
plt.xlabel("Dataset Size (MB)")
plt.ylabel("Execution Time (ms)")
plt.title("Execution Time versus Dataset Size")
#plt.title("CPU's Execution Time versus Dataset Size")
#plt.title("Kernel0's Execution Time versus Dataset Size")
plt.show()  
  

plt.figure()
