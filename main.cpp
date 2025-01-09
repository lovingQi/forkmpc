#include "j_mpc_flt.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <limits>

// 生成参考轨迹（直线）
std::vector<Eigen::VectorXd> generateReferencePath() {
    std::vector<Eigen::VectorXd> reference_path;
    
    // 生成一条更长的直线轨迹 (y = 0)
    double start_x = -10.0;  // 起点更远
    double end_x = 30.0;     // 终点更远
    int points = 200;        // 增加轨迹点数
    
    for(int i = 0; i < points; i++) {
        Eigen::VectorXd state(3);
        
        // x坐标从start_x到end_x均匀分布
        double x = start_x + (end_x - start_x) * i / (points - 1);
        
        // 位置 (x, 0)
        state(0) = x;      // x
        state(1) = 0.0;    // y = 0 (直线在x轴上)
        state(2) = 0.0;    // phi = 0 (航向角与x轴平行)
        
        reference_path.push_back(state);
    }
    
    return reference_path;
}

// 修改找最近点的函数，增加预瞄距离
int findClosestPoint(const Eigen::VectorXd& current_state, 
                    const std::vector<Eigen::VectorXd>& reference_path,
                    double preview_distance = 3.0) {  // 增加预瞄距离参数
    int closest_idx = 0;
    double min_dist = std::numeric_limits<double>::max();
    
    // 1. 找到最近点
    for(int i = 0; i < reference_path.size(); i++) {
        double dx = current_state(0) - reference_path[i](0);
        double dy = current_state(1) - reference_path[i](1);
        double dist = dx*dx + dy*dy;
        
        if(dist < min_dist) {
            min_dist = dist;
            closest_idx = i;
        }
    }
    
    // 2. 从最近点往前找预瞄点
    double accumulated_dist = 0.0;
    int preview_idx = closest_idx;
    
    while(preview_idx < reference_path.size() - 1 && 
          accumulated_dist < preview_distance) {
        double dx = reference_path[preview_idx + 1](0) - reference_path[preview_idx](0);
        double dy = reference_path[preview_idx + 1](1) - reference_path[preview_idx](1);
        accumulated_dist += std::sqrt(dx*dx + dy*dy);
        preview_idx++;
    }
    
    return preview_idx;
}

// 修改预瞄距离计算
double getPreviewDistance(double velocity) {
    const double base_preview = 2.0;  // 基础预瞄距离
    const double velocity_gain = 2.0;  // 速度增益
    return base_preview + velocity_gain * std::abs(velocity);  // 预瞄距离随速度增加
}

int main() {
    // 创建文件保存轨迹
    std::ofstream trajectory_file("trajectory.csv");
    trajectory_file << "x,y,theta,speed,steer\n";  // 修改CSV头以匹配实际列名
    
    // 1. 创建MPC控制器
    JMpcFlt mpc;
    
    // 2. 生成参考轨迹
    std::vector<Eigen::VectorXd> reference_path = generateReferencePath();
    
    // 3. 设置初始状态和控制
    Eigen::VectorXd current_state(3);
    current_state << 0.0, 3.0, -M_PI/6;  // 增大初始横向偏差到3米
    
    Eigen::VectorXd last_control(2);
    last_control << 0.0, 0.0;  // 初始速度和转向角都为0
    
    // 4. 模拟控制过程
    int sim_steps = 200;  // 模拟步数
    
    std::cout << "开始模拟..." << std::endl;
    for(int i = 0; i < sim_steps; i++) {
        double preview_dist = getPreviewDistance(last_control(0));
        int preview_idx = findClosestPoint(current_state, reference_path, preview_dist);
        
        // 获取从预瞄点开始往前的预测范围内的参考轨迹
        std::vector<Eigen::VectorXd> local_reference;
        for(int j = 0; j < mpc.Np; j++) {
            int idx = std::min(preview_idx + j, (int)reference_path.size() - 1);
            local_reference.push_back(reference_path[idx]);
        }
        
        // 计算控制输入
        Eigen::VectorXd control = mpc.solve(current_state, local_reference, last_control);
        
        // 打印当前状态和控制量
        std::cout << "Step " << i << ":" << std::endl;
        std::cout << "Position: x=" << current_state(0) 
                 << ", y=" << current_state(1)
                 << ", phi=" << current_state(2) << std::endl;
        std::cout << "Control: v=" << control(0)
                 << ", delta=" << control(1) << std::endl;
        std::cout << "-------------------" << std::endl;
        
        // 保存状态和控制量到文件
        trajectory_file << current_state(0) << "," 
                        << current_state(1) << "," 
                        << current_state(2) << ","
                        << control(0) << ","
                        << control(1) << "\n";
        
        // 更新状态（简单欧拉积分）
        double dt = 0.1;  // 与MPC中的dt保持一致
        double v = control(0);
        double delta = control(1);
        double phi = current_state(2);
        
        current_state(0) += v * cos(phi) * dt;
        current_state(1) += v * sin(phi) * dt;
        current_state(2) += v * tan(delta) / 2.7 * dt;  // 2.7是轴距
        
        // 更新上一时刻控制量
        last_control = control;
    }
    
    trajectory_file.close();
    std::cout << "模拟完成!" << std::endl;
    
    return 0;
} 