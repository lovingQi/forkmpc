#include "mpc_forklift.h"
#include <iostream>

int main() {
    // 初始化MPC控制器
    double dt = 0.1;          // 采样时间0.1秒
    double L = 1.0;           // 轴距1米
    int pred_horizon = 20;    // 预测时域20步
    double max_steer = M_PI/4;// 最大转向角45度
    double max_speed = 1.0;   // 最大速度1m/s
    
    MPCController mpc(dt, L, pred_horizon, max_steer, max_speed);
    
    // 当前状态
    Eigen::Vector3d current_state(0, 0, 0);  // [x, y, theta]
    
    // 参考路径（示例：直线路径）
    std::vector<Eigen::Vector2d> ref_path;
    for (int i = 0; i < pred_horizon; ++i) {
        ref_path.push_back(Eigen::Vector2d(i * dt, 0));
    }
    
    // 计算控制输入
    double steer, speed;
    if (mpc.solve(current_state, ref_path, steer, speed)) {
        std::cout << "计算成功！" << std::endl;
        std::cout << "转向角: " << steer << " rad" << std::endl;
        std::cout << "速度: " << speed << " m/s" << std::endl;
    } else {
        std::cout << "MPC求解失败！" << std::endl;
    }
    
    return 0;
} 