import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# 读取轨迹数据
df = pd.read_csv('trajectory.csv')

# 创建图形
fig = plt.figure(figsize=(15, 10))

# 设置子图
ax1 = plt.subplot(2, 2, 1)  # 轨迹
ax2 = plt.subplot(2, 2, 2)  # 航向角
ax3 = plt.subplot(2, 2, 3)  # 速度
ax4 = plt.subplot(2, 2, 4)  # 转向角

# 转换数据为numpy数组
x = df['x'].to_numpy()
y = df['y'].to_numpy()
theta = df['theta'].to_numpy()
speed = df['speed'].to_numpy()
steer = df['steer'].to_numpy()
time = np.arange(len(df)) * 0.1

# 绘制参考轨迹（固定的）
x_ref = np.linspace(-10, 50, 200)  # 延长参考轨迹显示范围
y_ref = np.zeros_like(x_ref)
ax1.plot(x_ref, y_ref, 'r--', label='Reference Path')
ax1.grid(True)
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_title('Vehicle Trajectory')
ax1.axis('equal')
ax1.set_xlim(-12, 52)  # 调整显示范围
ax1.set_ylim(-4, 4)

# 初始化动态线条
line_traj, = ax1.plot([], [], 'b-', label='Actual Trajectory')
point_vehicle, = ax1.plot([], [], 'go', markersize=10, label='Vehicle')
arrow_vehicle = ax1.arrow(0, 0, 0, 0, head_width=0.1, head_length=0.2, fc='g', ec='g')

line_heading, = ax2.plot([], [], 'g-')
line_speed, = ax3.plot([], [], 'r-')
line_steer, = ax4.plot([], [], 'b-')

# 设置其他子图的属性和范围
ax2.grid(True)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Heading (rad)')
ax2.set_title('Heading Angle')
ax2.set_ylim(min(theta) - 0.1, max(theta) + 0.1)  # 设置合适的显示范围

ax3.grid(True)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Velocity (m/s)')
ax3.set_title('Vehicle Speed')
ax3.set_ylim(min(speed) - 0.1, max(speed) + 0.1)  # 设置合适的显示范围

ax4.grid(True)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Steering Angle (rad)')
ax4.set_title('Steering Angle')
ax4.set_ylim(min(steer) - 0.1, max(steer) + 0.1)  # 设置合适的显示范围

# 添加当前值显示
text_head = ax2.text(0.02, 0.95, '', transform=ax2.transAxes)
text_speed = ax3.text(0.02, 0.95, '', transform=ax3.transAxes)
text_steer = ax4.text(0.02, 0.95, '', transform=ax4.transAxes)

ax1.legend()

def init():
    line_traj.set_data([], [])
    point_vehicle.set_data([], [])
    line_heading.set_data([], [])
    line_speed.set_data([], [])
    line_steer.set_data([], [])
    return line_traj, point_vehicle, line_heading, line_speed, line_steer

def update(frame):
    # 更新轨迹
    line_traj.set_data(x[:frame], y[:frame])
    point_vehicle.set_data(x[frame], y[frame])
    
    # 更新车辆方向箭头
    if hasattr(update, 'arrow'):
        update.arrow.remove()
    arrow_length = 0.5
    dx = arrow_length * np.cos(theta[frame])
    dy = arrow_length * np.sin(theta[frame])
    update.arrow = ax1.arrow(x[frame], y[frame], dx, dy,
                           head_width=0.1, head_length=0.2, 
                           fc='g', ec='g')
    
    # 更新其他图表
    line_heading.set_data(time[:frame], theta[:frame])
    line_speed.set_data(time[:frame], speed[:frame])
    line_steer.set_data(time[:frame], steer[:frame])
    
    # 更新数值显示
    text_head.set_text(f'Current: {theta[frame]:.2f} rad')
    text_speed.set_text(f'Current: {speed[frame]:.2f} m/s')
    text_steer.set_text(f'Current: {steer[frame]:.2f} rad')
    
    # 添加垂直线表示当前时间点
    for ax in [ax2, ax3, ax4]:
        if hasattr(ax, 'time_line'):
            ax.time_line.remove()
        ax.time_line = ax.axvline(time[frame], color='r', linestyle='--', alpha=0.5)
    
    return line_traj, point_vehicle, line_heading, line_speed, line_steer

# 创建动画，降低interval使动画更流畅
anim = FuncAnimation(fig, update, frames=len(df), 
                    init_func=init, interval=20,  # 20ms per frame
                    blit=False, repeat=True)  # 允许重复播放

plt.tight_layout()
plt.show() 