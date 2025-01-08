#include "mpc_forklift.h"
#include <iostream>
#include <cmath>
#include <fstream>

// 生成圆形参考路径
std::vector<Eigen::Vector2d> generateCirclePath(double radius, int points) {
    std::vector<Eigen::Vector2d> path;
    for (int i = 0; i < points; ++i) {
        double angle = 2.0 * M_PI * i / points;
        path.push_back(Eigen::Vector2d(
            radius * cos(angle),
            radius * sin(angle)
        ));
    }
    return path;
}

int main() {
    // 初始化MPC控制器
    double dt = 0.1;          // 采样时间0.1秒
    double L = 1.0;           // 轴距1米
    int pred_horizon = 20;    // 预测时域20步
    double max_steer = M_PI/4;// 最大转向角45度
    double max_speed = 1.0;   // 最大速度1m/s
    
    MPCController mpc(dt, L, pred_horizon, max_steer, max_speed);
    
    // 生成圆形参考路径
    double radius = 5.0;  // 5米半径
    auto ref_path = generateCirclePath(radius, 100);
    
    // 初始状态
    Eigen::Vector3d state(radius, 0, 0);  // 从圆上的一点开始
    
    // 用于保存轨迹
    std::ofstream trajectory_file("trajectory.csv");
    trajectory_file << "x,y,theta,steer,speed\n";
    
    // 仿真循环
    for (int i = 0; i < 200; ++i) {  // 模拟20秒
        // 计算控制输入
        double steer, speed;
        if (mpc.solve(state, ref_path, steer, speed)) {
            // 记录当前状态
            trajectory_file << state[0] << "," << state[1] << "," 
                          << state[2] << "," << steer << "," << speed << "\n";
            
            // 更新状态（简单运动学模型）
            state[0] += speed * cos(state[2]) * cos(steer) * dt;
            state[1] += speed * sin(state[2]) * cos(steer) * dt;
            state[2] += speed * sin(steer) / L * dt;
            
            std::cout << "Time: " << i*dt << "s, "
                     << "Position: (" << state[0] << ", " << state[1] << "), "
                     << "Heading: " << state[2] << ", "
                     << "Controls: [" << steer << ", " << speed << "]\n";
        } else {
            std::cout << "MPC求解失败！\n";
            break;
        }
    }
    
    trajectory_file.close();
    std::cout << "轨迹已保存到trajectory.csv\n";
    
    return 0;
} 