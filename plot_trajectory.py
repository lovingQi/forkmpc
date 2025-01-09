import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取轨迹数据
df = pd.read_csv('trajectory.csv')

# 打印列名以检查
print("CSV文件的列名:", df.columns.tolist())

# 创建图形
plt.figure(figsize=(15, 10))

# 绘制子图1：轨迹
plt.subplot(2, 2, 1)
# 将pandas数据转换为numpy数组
x = df['x'].to_numpy()
y = df['y'].to_numpy()
plt.plot(x, y, 'b-', label='Actual Trajectory')

# 绘制参考直线轨迹
x_ref = np.linspace(-5, 15, 100)
y_ref = np.zeros_like(x_ref)
plt.plot(x_ref, y_ref, 'r--', label='Reference Path')

# 绘制起点
plt.plot(x[0], y[0], 'go', markersize=10, label='Start Point')

plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Vehicle Trajectory')
plt.axis('equal')
plt.grid(True)
plt.legend()

# 调整显示范围
plt.xlim(-6, 16)
plt.ylim(-3, 3)

# 绘制子图2：航向角
plt.subplot(2, 2, 2)
time = np.arange(len(df)) * 0.1
plt.plot(time, df['theta'].to_numpy(), 'g-')
plt.xlabel('Time (s)')
plt.ylabel('Heading (rad)')
plt.title('Heading Angle')
plt.grid(True)

# 绘制子图3：速度
plt.subplot(2, 2, 3)
plt.plot(time, df['speed'].to_numpy(), 'r-')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Vehicle Speed')
plt.grid(True)

# 绘制子图4：转向角
plt.subplot(2, 2, 4)
plt.plot(time, df['steer'].to_numpy(), 'b-')
plt.xlabel('Time (s)')
plt.ylabel('Steering Angle (rad)')
plt.title('Steering Angle')
plt.grid(True)

plt.tight_layout()
plt.savefig('mpc_results.png')
plt.show() 