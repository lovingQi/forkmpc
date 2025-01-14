import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

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
point_reference, = ax1.plot([], [], 'ro', markersize=8, label='Reference Point')  # 添加参考点
preview_line, = ax1.plot([], [], 'g--', alpha=0.5, label='Preview Line')  # 添加预瞄线
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

# 添加叉车车体绘制函数
def draw_forklift(ax, x, y, theta, steer_angle, color='g'):
    # 车体参数（单位：米）
    length = 2.7  # 车长
    width = 1.0   # 车宽
    wheel_radius = 0.2  # 车轮半径
    
    # 计算车体四个角的位置（相对于车体中心）
    corners = np.array([
        [-length/2, -width/2],
        [length/2, -width/2],
        [length/2, width/2],
        [-length/2, width/2]
    ])
    
    # 旋转矩阵
    rot = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    # 转换到全局坐标系
    corners = np.dot(corners, rot.T)
    corners = corners + np.array([x, y])
    
    # 绘制车体
    if hasattr(draw_forklift, 'body'):
        draw_forklift.body.remove()
    draw_forklift.body = patches.Polygon(corners, color=color, alpha=0.3)
    ax.add_patch(draw_forklift.body)
    
    # 绘制舵轮（前轮）
    front_pos = np.dot([length/2, 0], rot.T) + np.array([x, y])
    if hasattr(draw_forklift, 'wheel'):
        draw_forklift.wheel.remove()
    
    # 舵轮形状
    wheel_length = 0.4
    wheel_width = 0.1
    wheel_corners = np.array([
        [-wheel_length/2, -wheel_width/2],
        [wheel_length/2, -wheel_width/2],
        [wheel_length/2, wheel_width/2],
        [-wheel_length/2, wheel_width/2]
    ])
    
    # 舵轮旋转矩阵（考虑车体方向和转向角）
    wheel_rot = np.array([
        [np.cos(theta + steer_angle), -np.sin(theta + steer_angle)],
        [np.sin(theta + steer_angle), np.cos(theta + steer_angle)]
    ])
    
    wheel_corners = np.dot(wheel_corners, wheel_rot.T)
    wheel_corners = wheel_corners + front_pos
    
    draw_forklift.wheel = patches.Polygon(wheel_corners, color='blue', alpha=0.8)
    ax.add_patch(draw_forklift.wheel)
    
    # 绘制后轮（固定轮）
    rear_pos = np.dot([-length/2, 0], rot.T) + np.array([x, y])
    if hasattr(draw_forklift, 'rear_wheel'):
        draw_forklift.rear_wheel.remove()
    
    rear_wheel_corners = np.array([
        [-wheel_length/2, -wheel_width/2],
        [wheel_length/2, -wheel_width/2],
        [wheel_length/2, wheel_width/2],
        [-wheel_length/2, wheel_width/2]
    ])
    
    rear_wheel_corners = np.dot(rear_wheel_corners, rot.T)
    rear_wheel_corners = rear_wheel_corners + rear_pos
    
    draw_forklift.rear_wheel = patches.Polygon(rear_wheel_corners, color='black', alpha=0.8)
    ax.add_patch(draw_forklift.rear_wheel)

def init():
    line_traj.set_data([], [])
    point_vehicle.set_data([], [])
    point_reference.set_data([], [])  # 初始化参考点
    preview_line.set_data([], [])     # 初始化预瞄线
    line_heading.set_data([], [])
    line_speed.set_data([], [])
    line_steer.set_data([], [])
    return line_traj, point_vehicle, point_reference, preview_line, line_heading, line_speed, line_steer

def update(frame):
    # 更新轨迹
    line_traj.set_data(x[:frame], y[:frame])
    point_vehicle.set_data(x[frame], y[frame])
    
    # 计算最近参考点
    current_pos = np.array([x[frame], y[frame]])
    ref_points = np.column_stack((x_ref, y_ref))
    distances = np.linalg.norm(ref_points - current_pos, axis=1)
    closest_idx = np.argmin(distances)
    
    # 计算预瞄点（基于当前速度）
    preview_dist = 2.0 + 2.0 * abs(speed[frame])  # 基础预瞄距离 + 速度相关预瞄距离
    preview_idx = closest_idx
    accumulated_dist = 0.0
    while preview_idx < len(x_ref) - 1 and accumulated_dist < preview_dist:
        dx = x_ref[preview_idx + 1] - x_ref[preview_idx]
        dy = y_ref[preview_idx + 1] - y_ref[preview_idx]
        accumulated_dist += np.sqrt(dx*dx + dy*dy)
        preview_idx += 1
    
    # 更新参考点显示
    point_reference.set_data(x_ref[preview_idx], y_ref[preview_idx])
    
    # 更新预瞄线（从车辆到预瞄点）
    preview_line.set_data([x[frame], x_ref[preview_idx]], 
                         [y[frame], y_ref[preview_idx]])
    
    # 绘制叉车车体和舵轮
    draw_forklift(ax1, x[frame], y[frame], theta[frame], steer[frame])
    
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
    
    return line_traj, point_vehicle, point_reference, preview_line, line_heading, line_speed, line_steer

# 创建动画，降低interval使动画更流畅
anim = FuncAnimation(fig, update, frames=len(df), 
                    init_func=init, interval=20,  # 20ms per frame
                    blit=False, repeat=True)  # 允许重复播放

plt.tight_layout()
plt.show() 