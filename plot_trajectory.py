import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

# 读取轨迹数据
df = pd.read_csv('trajectory.csv')

# 创建图形
fig = plt.figure(figsize=(15, 18))  # 增加图形高度以容纳6张图

# 设置子图
ax1 = plt.subplot(3, 2, 1)  # 轨迹
ax2 = plt.subplot(3, 2, 2)  # 航向角
ax3 = plt.subplot(3, 2, 3)  # 速度
ax4 = plt.subplot(3, 2, 4)  # 转向角
ax5 = plt.subplot(3, 2, 5)  # lateral error
ax6 = plt.subplot(3, 2, 6)  # heading error

# 转换数据为numpy数组
x = df['x'].to_numpy()
y = df['y'].to_numpy()
theta = df['theta'].to_numpy()
speed = df['speed'].to_numpy()
steer = df['steer'].to_numpy()
time = np.arange(len(df)) * 0.03
def cubic_bezier(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    mt = 1 - t
    mt2 = mt * mt
    mt3 = mt2 * mt
    return p0 * mt3 + p1 * (3 * mt2 * t) + p2 * (3 * mt * t2) + p3 * t3

# 生成参考路径点
density = 167
x_ref = []
y_ref = []

# 第一段：直线
t = np.linspace(0, 1, density)
x_ref.extend(-10 + t * 10)
y_ref.extend(np.zeros_like(t))

# 第二段：右转弯（贝塞尔曲线）
p0 = np.array([0, 0])
p1 = np.array([5, 0])
p2 = np.array([10, 2])
p3 = np.array([10, 5])

for t in np.linspace(0, 1, density):
    point = cubic_bezier(p0, p1, p2, p3, t)
    x_ref.append(point[0])
    y_ref.append(point[1])

# 第三段：直线
t = np.linspace(0, 1, density)
x_ref.extend(np.full_like(t, 10))
y_ref.extend(5 + t * 5)

# 第四段：左转弯
p0 = np.array([10, 10])
p1 = np.array([10, 13])
p2 = np.array([8, 15])
p3 = np.array([5, 15])

for t in np.linspace(0, 1, density):
    point = cubic_bezier(p0, p1, p2, p3, t)
    x_ref.append(point[0])
    y_ref.append(point[1])

# 第五段：直线
t = np.linspace(0, 1, density)
x_ref.extend(5 - t * 15)
y_ref.extend(np.full_like(t, 15))

# 在生成参考路径后，计算参考航向角
theta_ref = np.zeros_like(x_ref)
for i in range(1, len(x_ref)-1):
    dx = x_ref[i+1] - x_ref[i-1]
    dy = y_ref[i+1] - y_ref[i-1]
    theta_ref[i] = np.arctan2(dy, dx)

# 处理首尾点的航向角
theta_ref[0] = theta_ref[1]
theta_ref[-1] = theta_ref[-2]

# 转换为numpy数组
x_ref = np.array(x_ref)
y_ref = np.array(y_ref)
theta_ref = np.array(theta_ref)  # 确保theta_ref也是numpy数组
# 绘制参考轨迹（固定的）

ax1.plot(x_ref, y_ref, 'r--', label='Reference Path')
ax1.grid(True)
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_title('Vehicle Trajectory')
ax1.axis('equal')
ax1.set_xlim(-15, 15)
ax1.set_ylim(-5, 20)

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
ax2.set_ylabel('Heading (deg)')
ax2.set_title('Heading Angle')
ax2.set_ylim(np.degrees(min(theta)) - 5, np.degrees(max(theta)) + 5)

ax3.grid(True)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Velocity (m/s)')
ax3.set_title('Vehicle Speed')
ax3.set_ylim(min(speed) - 0.1, max(speed) + 0.1)

ax4.grid(True)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Steering Angle (deg)')
ax4.set_title('Steering Angle')
ax4.set_ylim(np.degrees(min(steer)) - 5, np.degrees(max(steer)) + 5)

# 添加当前值显示
text_head = ax2.text(0.02, 0.95, '', transform=ax2.transAxes)
text_speed = ax3.text(0.02, 0.95, '', transform=ax3.transAxes)
text_steer = ax4.text(0.02, 0.95, '', transform=ax4.transAxes)

ax1.legend()

# 添加叉车车体绘制函数
def draw_forklift(ax, x, y, theta, steer_angle, color='g'):
    # 车体参数（单位：米）
    length = 1.4     # 车长
    width = 0.6      # 车宽
    wheelbase = 0.97 # 轴距（后轮到前轮的距离）
    rear_overhang = 0.3  # 后悬（后轮到车尾的距离）
    
    # 计算后轮位置（当前x,y是后轮位置）
    rear_pos = np.array([x, y])
    
    # 计算车体中心位置（从后轮向前wheelbase/2，再向后rear_overhang/2）
    center_offset = (wheelbase/2 - rear_overhang/2)
    center_x = x + center_offset * np.cos(theta)
    center_y = y + center_offset * np.sin(theta)
    
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
    corners = corners + np.array([center_x, center_y])
    
    # 绘制车体
    if hasattr(draw_forklift, 'body'):
        draw_forklift.body.remove()
    draw_forklift.body = patches.Polygon(corners, color=color, alpha=0.3)
    ax.add_patch(draw_forklift.body)
    
    # 绘制前轮（转向轮）
    front_x = x + wheelbase * np.cos(theta)
    front_y = y + wheelbase * np.sin(theta)
    
    if hasattr(draw_forklift, 'wheel'):
        draw_forklift.wheel.remove()
    
    # 前轮形状
    wheel_length = 0.2
    wheel_width = 0.08
    wheel_corners = np.array([
        [-wheel_length/2, -wheel_width/2],
        [wheel_length/2, -wheel_width/2],
        [wheel_length/2, wheel_width/2],
        [-wheel_length/2, wheel_width/2]
    ])
    
    # 前轮旋转矩阵（考虑车体方向和转向角）
    wheel_rot = np.array([
        [np.cos(theta + steer_angle), -np.sin(theta + steer_angle)],
        [np.sin(theta + steer_angle), np.cos(theta + steer_angle)]
    ])
    
    wheel_corners = np.dot(wheel_corners, wheel_rot.T)
    wheel_corners = wheel_corners + np.array([front_x, front_y])
    
    draw_forklift.wheel = patches.Polygon(wheel_corners, color='blue', alpha=0.8)
    ax.add_patch(draw_forklift.wheel)
    
    # 绘制后轮
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
    preview_dist = 0.06+ 0.045 * abs(speed[frame])  # 修改为与C++相同的参数
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
    
    # 更新其他图表（转换为度）
    line_heading.set_data(time[:frame], np.degrees(theta[:frame]))
    line_speed.set_data(time[:frame], speed[:frame])
    line_steer.set_data(time[:frame], np.degrees(steer[:frame]))
    
    # 更新数值显示（转换为度）
    text_head.set_text(f'Current: {np.degrees(theta[frame]):.1f}°')
    text_speed.set_text(f'Current: {speed[frame]:.2f} m/s')
    text_steer.set_text(f'Current: {np.degrees(steer[frame]):.1f}°')
    
    # 计算误差
    lateral_errors = []
    heading_errors = []
    for i in range(frame + 1):
        # 找到最近参考点
        current_pos = np.array([x[i], y[i]])
        ref_points = np.column_stack((x_ref, y_ref))
        distances = np.linalg.norm(ref_points - current_pos, axis=1)
        closest_idx = np.argmin(distances)
        
        # 计算参考线方向
        path_angle = theta_ref[closest_idx]  # 直接使用参考点的航向角
        
        # 计算横向误差 (转换为毫米)
        lateral_error = abs((y[i] - y_ref[closest_idx]) * np.cos(path_angle) - 
                          (x[i] - x_ref[closest_idx]) * np.sin(path_angle)) * 1000
        
        # 计算航向误差 (转换为度)
        heading_error = abs(theta[i] - path_angle)
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        heading_error = np.degrees(heading_error)
        
        lateral_errors.append(lateral_error)
        heading_errors.append(heading_error)
    
    # 更新横向误差图
    ax5.clear()
    ax5.grid(True)
    ax5.plot(time[:frame+1], lateral_errors, 'b-')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Lateral Error (mm)')
    ax5.set_title('Lateral Tracking Error')
    if frame > 0:
        ax5.text(0.02, 0.95, f'Current: {lateral_errors[-1]:.1f} mm', 
                transform=ax5.transAxes)
    
    # 更新航向误差图
    ax6.clear()
    ax6.grid(True)
    ax6.plot(time[:frame+1], heading_errors, 'r-')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Heading Error (deg)')
    ax6.set_title('Heading Tracking Error')
    if frame > 0:
        ax6.text(0.02, 0.95, f'Current: {heading_errors[-1]:.1f}°', 
                transform=ax6.transAxes)
    
    # 添加垂直线表示当前时间点
    for ax in [ax2, ax3, ax4, ax5, ax6]:
        if hasattr(ax, 'time_line'):
            ax.time_line.remove()
        ax.time_line = ax.axvline(time[frame], color='r', linestyle='--', alpha=0.5)
    
    return line_traj, point_vehicle, point_reference, preview_line, line_heading, line_speed, line_steer

# 创建动画，降低interval使动画更流畅
anim = FuncAnimation(fig, update, frames=len(df), 
                    init_func=init, interval=5,  # 从20ms改为5ms
                    blit=False, repeat=True)

plt.tight_layout()  # 调整子图间距
plt.show() 