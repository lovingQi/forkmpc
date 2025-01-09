#include "j_mpc_flt.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <limits>

// 添加贝塞尔曲线计算函数
Eigen::Vector2d cubicBezier(const Eigen::Vector2d& p0, const Eigen::Vector2d& p1,
                           const Eigen::Vector2d& p2, const Eigen::Vector2d& p3, double t) {
    double t2 = t * t;
    double t3 = t2 * t;
    double mt = 1 - t;
    double mt2 = mt * mt;
    double mt3 = mt2 * mt;
    
    return p0 * mt3 + p1 * (3 * mt2 * t) + p2 * (3 * mt * t2) + p3 * t3;
}

// 生成复合参考路径
std::vector<Eigen::VectorXd> generateReferencePath() {
    std::vector<Eigen::VectorXd> reference_path;
    std::vector<Eigen::Vector2d> points;
    
    // 路径点密度
    const int density = 50;  // 每段曲线的采样点数
    
    // 第一段：直线
    for(int i = 0; i < density; i++) {
        double t = static_cast<double>(i) / density;
        Eigen::Vector2d point(-10 + t * 10, 0);  // 从(-10,0)到(0,0)的直线
        points.push_back(point);
    }
    
    // 第二段：右转弯（贝塞尔曲线）
    Eigen::Vector2d p0(0, 0);
    Eigen::Vector2d p1(5, 0);
    Eigen::Vector2d p2(10, 2);
    Eigen::Vector2d p3(10, 5);
    
    for(int i = 0; i < density; i++) {
        double t = static_cast<double>(i) / density;
        points.push_back(cubicBezier(p0, p1, p2, p3, t));
    }
    
    // 第三段：直线
    for(int i = 0; i < density; i++) {
        double t = static_cast<double>(i) / density;
        Eigen::Vector2d point(10, 5 + t * 5);  // 向上的直线
        points.push_back(point);
    }
    
    // 第四段：左转弯
    p0 = Eigen::Vector2d(10, 10);
    p1 = Eigen::Vector2d(10, 13);
    p2 = Eigen::Vector2d(8, 15);
    p3 = Eigen::Vector2d(5, 15);
    
    for(int i = 0; i < density; i++) {
        double t = static_cast<double>(i) / density;
        points.push_back(cubicBezier(p0, p1, p2, p3, t));
    }
    
    // 第五段：直线
    for(int i = 0; i < density; i++) {
        double t = static_cast<double>(i) / density;
        Eigen::Vector2d point(5 - t * 15, 15);  // 向左的直线
        points.push_back(point);
    }
    
    // 计算路径点的航向角
    for(size_t i = 0; i < points.size(); i++) {
        Eigen::VectorXd state(3);
        state(0) = points[i].x();
        state(1) = points[i].y();
        
        // 计算航向角（使用前后点计算切线方向）
        if(i == 0) {
            double dx = points[1].x() - points[0].x();
            double dy = points[1].y() - points[0].y();
            state(2) = std::atan2(dy, dx);
        }
        else if(i == points.size() - 1) {
            double dx = points[i].x() - points[i-1].x();
            double dy = points[i].y() - points[i-1].y();
            state(2) = std::atan2(dy, dx);
        }
        else {
            double dx = points[i+1].x() - points[i-1].x();
            double dy = points[i+1].y() - points[i-1].y();
            state(2) = std::atan2(dy, dx);
        }
        
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
double getPreviewDistance(double velocity, double lateral_error, bool is_curve) {
    const double base_preview = is_curve ? 0.5 : 1.0;   // 曲线段时基础预瞄距离更短
    const double velocity_gain = is_curve ? 0.3 : 0.6;  // 曲线段时速度增益更小
    
    // 预瞄距离在误差大时更短，在曲线段时显著缩短
    double preview = base_preview + 
                    velocity_gain * std::abs(velocity) * std::exp(-1.0 * lateral_error);
    
    // 限制预瞄距离的范围，曲线段时范围更小
    return is_curve ? 
           std::max(0.3, std::min(preview, 1.0)) :  // 曲线段
           std::max(0.5, std::min(preview, 2.0));   // 直线段
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
    current_state << -10.0, 2.0, 0.0;  // 起点位置和航向角
    
    Eigen::VectorXd last_control(2);
    last_control << 0.0, 0.0;  // 初始速度和转向角都为0
    
    // 4. 模拟控制过程
    int sim_steps = 800;  // 增加仿真步数以完成更长的路径
    
    std::cout << "开始模拟..." << std::endl;
    for(int i = 0; i < sim_steps; i++) {
        // 检测是否在曲线段
        bool is_curve = false;
        if (preview_idx + 1 < reference_path.size()) {
            double dx = reference_path[preview_idx+1](0) - reference_path[preview_idx](0);
            double dy = reference_path[preview_idx+1](1) - reference_path[preview_idx](1);
            double dphi = reference_path[preview_idx+1](2) - reference_path[preview_idx](2);
            double ds = std::sqrt(dx*dx + dy*dy);
            double curvature = std::abs(dphi) / (ds + 1e-6);
            is_curve = (curvature > 0.1);  // 曲率阈值可以调整
        }

        double lateral_error = std::abs(current_state(1));
        double preview_dist = getPreviewDistance(last_control(0), lateral_error, is_curve);
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