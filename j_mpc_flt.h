#ifndef J_MPC_FLT_H
#define J_MPC_FLT_H

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include <vector>
#include <osqp/osqp.h>

// 在类定义前添加结构体声明

class JMpcFlt {
public:
    JMpcFlt();
    
    // 基本参数定义
    static const int STATE_DIM = 3;     // [x, y, phi]
    static const int CONTROL_DIM = 2;   // [v, delta]
    static const int AUG_STATE_DIM = STATE_DIM + CONTROL_DIM;  // 增广状态维度
    static const int Np = 10;           // 预测步数
    static const int Nc = 9;           // 控制步数
    
    // 主要接口
    Eigen::VectorXd solve(const Eigen::VectorXd& current_state,
                         const std::vector<Eigen::VectorXd>& reference_path,
                         const Eigen::VectorXd& last_control);
    struct TrackingErrors {
    double lateral_error;
    double heading_error;
};
                         
private:
    // 系统参数
    double dt_;        // 采样时间
    double max_delta_; // 最大转向角
    double L_;         // 轴距
    double max_v_;     // 最大速度
    double rho_;       // 松弛因子权重
    
    // 权重矩阵
    Eigen::MatrixXd Q_;  // 状态误差权重
    Eigen::MatrixXd R_;  // 控制增量权重
    
    // 辅助函数 - Kronecker积
    Eigen::MatrixXd kroneckerProduct(const Eigen::MatrixXd& a, 
                                    const Eigen::MatrixXd& b) const {
        Eigen::MatrixXd result(a.rows()*b.rows(), a.cols()*b.cols());
        for(int i = 0; i < a.rows(); i++)
            for(int j = 0; j < a.cols(); j++)
                result.block(i*b.rows(), j*b.cols(), b.rows(), b.cols()) = a(i,j)*b;
        return result;
    }
    
    // 1. 计算线性化模型 (步骤2-3)
    void linearizeModel(const Eigen::VectorXd& reference_state,
                       const Eigen::VectorXd& reference_control,
                       Eigen::MatrixXd& A,
                       Eigen::MatrixXd& B) const;
                       
    // 2. 构建增广系统矩阵 (步骤8)
    void buildAugmentedSystem(const Eigen::MatrixXd& A,
                             const Eigen::MatrixXd& B,
                             Eigen::MatrixXd& A_tilde,
                             Eigen::MatrixXd& B_tilde,
                             Eigen::MatrixXd& C_tilde);
                             
    // 3. 构建预测矩阵 (步骤9)
    void buildPredictionMatrices(const std::vector<Eigen::MatrixXd>& A_tilde_seq,
                                const std::vector<Eigen::MatrixXd>& B_tilde_seq,
                                const std::vector<Eigen::MatrixXd>& C_tilde_seq,
                                Eigen::MatrixXd& Psi,
                                Eigen::MatrixXd& Theta);
    
    // 4. 构建QP问题 (步骤10)
    void buildQPProblem(const Eigen::VectorXd& current_state,
                       const Eigen::VectorXd& last_control,
                       const std::vector<Eigen::VectorXd>& reference_path,
                       const Eigen::MatrixXd& Psi,
                       const Eigen::MatrixXd& Theta,
                       Eigen::SparseMatrix<double>& H,
                       Eigen::VectorXd& g,
                       Eigen::VectorXd& lb,
                       Eigen::VectorXd& ub);
                       
    // 添加函数声明
    TrackingErrors calculateTrackingErrors(
        const Eigen::VectorXd& current_state,
        const Eigen::VectorXd& reference_state);
    
    Eigen::VectorXd last_reference_control_;  // 存储上一时刻的参考控制量
};
#endif // J_MPC_FLT_H
