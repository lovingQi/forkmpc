#ifndef MPC_NONLINEAR_VEHICLE_H
#define MPC_NONLINEAR_VEHICLE_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <osqp/osqp.h>

class JMpcFlt {
public:
    JMpcFlt();
    
    // 基本参数定义
    static const int STATE_DIM = 3;     // [x, y, theta]
    static const int CONTROL_DIM = 2;   // [V, delta]
    static const int HORIZON = 20;      // 预测时域
    
    // 主要接口
    Eigen::VectorXd solve(const Eigen::VectorXd& current_state,
                         const std::vector<Eigen::VectorXd>& reference_path);
                         
private:
    // 系统参数
    double dt_;        // 采样时间
    double max_delta_; // 最大转向角
    double L_;         // 轴距
    double max_v_;     // 最大速度
    
    // 权重矩阵
    Eigen::MatrixXd Q_;  // 状态误差权重
    Eigen::MatrixXd R_;  // 控制输入权重
    
    // 线性化模型
    Eigen::MatrixXd linearizeModel(const Eigen::VectorXd& reference_state,
                                  const Eigen::VectorXd& reference_control) const;
    
    // 构建QP问题
    void buildQPProblem(const Eigen::VectorXd& current_state,
                       const std::vector<Eigen::VectorXd>& reference_path,
                       Eigen::SparseMatrix<double>& P,
                       Eigen::VectorXd& q,
                       Eigen::SparseMatrix<double>& A,
                       Eigen::VectorXd& l,
                       Eigen::VectorXd& u);
};

#endif // MPC_NONLINEAR_VEHICLE_H 