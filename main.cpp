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
    // MPC参数
    double dt = 0.1;  // 采样时间
    double L = 2.0;   // 轴距
    int pred_horizon = 10;  // 缩短预测时域，使控制更积极
    double max_steer = 0.6;  // 最大转向角约35度
    double max_speed = 2.0;  // 增大最大速度
    
    MPCController mpc(dt, L, pred_horizon, max_steer, max_speed);
    
    // 初始状态 [x, y, theta]
    Eigen::Vector3d current_state(5.0, 0.0, M_PI/2);  // 从右侧开始，朝向上方
    
    // 生成参考路径（圆形）
    std::vector<Eigen::Vector2d> ref_path;
    double radius = 5.0;
    int num_points = 30;   // 增加路径点，使路径更平滑
    for (int i = 0; i < num_points; i++) {
        // 生成1/4圆弧，从π/2到0
        double theta = M_PI/2 - (M_PI/2) * i / (num_points-1);
        double x = radius * cos(theta);
        double y = radius * sin(theta);
        ref_path.push_back(Eigen::Vector2d(x, y));
    }
    
    // 存储轨迹
    std::ofstream file("trajectory.csv");
    file << "x,y,theta,steer,speed\n";
    
    // 模拟
    double sim_time = 20.0;  // 减小仿真时间
    int steps = sim_time / dt;
    
    for (int i = 0; i < steps; i++) {
        double steer, speed;
        if (!mpc.solve(current_state, ref_path, steer, speed)) {
            std::cout << "MPC求解失败！" << std::endl;
            break;
        }
        
        // 更新状态
        current_state[0] += speed * cos(current_state[2]) * cos(steer) * dt;
        current_state[1] += speed * sin(current_state[2]) * cos(steer) * dt;
        current_state[2] += speed * sin(steer) / L * dt;
        
        // 记录轨迹
        file << current_state[0] << "," << current_state[1] << "," 
             << current_state[2] << "," << steer << "," << speed << "\n";
    }
    
    file.close();
    std::cout << "轨迹已保存到trajectory.csv" << std::endl;
    
    return 0;
} 