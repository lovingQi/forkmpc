import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取轨迹数据
data = pd.read_csv('trajectory.csv')

# 转换数据为numpy数组
x = data['x'].to_numpy()
y = data['y'].to_numpy()
theta = data['theta'].to_numpy()
steer = data['steer'].to_numpy()
speed = data['speed'].to_numpy()

# 创建图形
plt.figure(figsize=(12, 8))

# 绘制参考路径（圆）
circle_theta = np.linspace(0, 2*np.pi, 100)
radius = 5.0
plt.plot(radius*np.cos(circle_theta), radius*np.sin(circle_theta), 'r--', label='Reference Path')

# 绘制实际轨迹
plt.plot(x, y, 'b-', label='Actual Trajectory')

# 绘制起点和终点
plt.plot(x[0], y[0], 'go', label='Start', markersize=10)
plt.plot(x[-1], y[-1], 'ro', label='End', markersize=10)

# 添加箭头表示车辆朝向
skip = 10  # 每隔10个点画一个箭头
for i in range(0, len(x), skip):
    plt.arrow(x[i], y[i], 
              0.3*np.cos(theta[i]), 
              0.3*np.sin(theta[i]),
              head_width=0.1, head_length=0.2, fc='g', ec='g', alpha=0.5)

# 设置图形属性
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.title('Forklift Path Tracking Performance')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')

# 添加控制输入的子图
plt.figure(figsize=(12, 4))

# 时间数组
time = np.arange(len(data)) * 0.1  # dt = 0.1s

# 绘制转向角
plt.subplot(1, 2, 1)
plt.plot(time, steer, 'b-')
plt.grid(True)
plt.title('Steering Angle')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')

# 绘制速度
plt.subplot(1, 2, 2)
plt.plot(time, speed, 'r-')
plt.grid(True)
plt.title('Speed')
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')

# 调整布局并显示
plt.tight_layout()
plt.show() 