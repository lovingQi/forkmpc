#include "j_mpc_flt.h" 
#include <iostream>

JMpcFlt::JMpcFlt() {
    // 初始化系统参数
    dt_ = 0.1;         
    max_delta_ = 0.7;  // 约40度
    L_ = 2.7;         
    max_v_ = 1.0;     
    rho_ = 1e3;       
    
    // 调整权重矩阵
    Q_ = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);
    Q_(0,0) = 10.0;    // x位置误差
    Q_(1,1) = 500.0;   // y位置误差（大幅增加横向跟踪权重）
    Q_(2,2) = 100.0;   // 航向角误差
    
    R_ = Eigen::MatrixXd::Identity(CONTROL_DIM, CONTROL_DIM);
    R_(0,0) = 1.0;     // 速度增量权重
    R_(1,1) = 0.1;     // 转向角增量权重（降低以允许更灵活的转向）
}

void JMpcFlt::linearizeModel(
    const Eigen::VectorXd& reference_state,
    const Eigen::VectorXd& reference_control,
    Eigen::MatrixXd& A,
    Eigen::MatrixXd& B) const {
    
    // 提取参考状态和控制
    double x_ref = reference_state(0);
    double y_ref = reference_state(1);
    double phi_ref = reference_state(2);
    double v_ref = reference_control(0);
    double delta_ref = reference_control(1);
    
    // 计算雅可比矩阵A = ∂f/∂x
    A = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);
    A(0,2) = -v_ref * sin(phi_ref);  // ∂ẋ/∂φ
    A(1,2) = v_ref * cos(phi_ref);   // ∂ẏ/∂φ
    
    // 计算雅可比矩阵B = ∂f/∂u
    B = Eigen::MatrixXd::Zero(STATE_DIM, CONTROL_DIM);
    B(0,0) = cos(phi_ref);           // ∂ẋ/∂v
    B(1,0) = sin(phi_ref);           // ∂ẏ/∂v
    B(2,0) = tan(delta_ref) / L_;    // ∂φ̇/∂v
    B(2,1) = v_ref / (L_ * pow(cos(delta_ref), 2));  // ∂φ̇/∂δ
    
    // 离散化
    A = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM) + dt_ * A;
    B = dt_ * B;
}

void JMpcFlt::buildAugmentedSystem(
    const Eigen::MatrixXd& A,
    const Eigen::MatrixXd& B,
    Eigen::MatrixXd& A_tilde,
    Eigen::MatrixXd& B_tilde,
    Eigen::MatrixXd& C_tilde) {
    
    // 构建增广系统矩阵 (步骤8)
    A_tilde = Eigen::MatrixXd::Zero(AUG_STATE_DIM, AUG_STATE_DIM);
    A_tilde.topLeftCorner(STATE_DIM, STATE_DIM) = A;
    A_tilde.topRightCorner(STATE_DIM, CONTROL_DIM) = B;
    A_tilde.bottomRightCorner(CONTROL_DIM, CONTROL_DIM) = 
        Eigen::MatrixXd::Identity(CONTROL_DIM, CONTROL_DIM);
    
    B_tilde = Eigen::MatrixXd::Zero(AUG_STATE_DIM, CONTROL_DIM);
    B_tilde.topRows(STATE_DIM) = B;
    B_tilde.bottomRows(CONTROL_DIM) = 
        Eigen::MatrixXd::Identity(CONTROL_DIM, CONTROL_DIM);
    
    C_tilde = Eigen::MatrixXd::Zero(STATE_DIM, AUG_STATE_DIM);
    C_tilde.leftCols(STATE_DIM) = 
        Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);
}

void JMpcFlt::buildPredictionMatrices(
    const std::vector<Eigen::MatrixXd>& A_tilde_seq,
    const std::vector<Eigen::MatrixXd>& B_tilde_seq,
    const std::vector<Eigen::MatrixXd>& C_tilde_seq,
    Eigen::MatrixXd& Psi,
    Eigen::MatrixXd& Theta) {
    
    // 构建预测矩阵 (步骤9)
    Psi = Eigen::MatrixXd::Zero(STATE_DIM * Np, AUG_STATE_DIM);
    Theta = Eigen::MatrixXd::Zero(STATE_DIM * Np, CONTROL_DIM * Nc);
    
    // 构建Psi矩阵
    Eigen::MatrixXd temp = Eigen::MatrixXd::Identity(AUG_STATE_DIM, AUG_STATE_DIM);
    for(int i = 0; i < Np; i++) {
        temp = A_tilde_seq[i] * temp;
        Psi.block(i*STATE_DIM, 0, STATE_DIM, AUG_STATE_DIM) = 
            C_tilde_seq[i] * temp;
    }
    
    // 构建Theta矩阵
    for(int i = 0; i < Np; i++) {
        for(int j = 0; j <= std::min(i, Nc-1); j++) {
            Eigen::MatrixXd temp = Eigen::MatrixXd::Identity(AUG_STATE_DIM, AUG_STATE_DIM);
            for(int k = i; k > j; k--) {
                temp = A_tilde_seq[k-1] * temp;
            }
            Theta.block(i*STATE_DIM, j*CONTROL_DIM, STATE_DIM, CONTROL_DIM) = 
                C_tilde_seq[i] * temp * B_tilde_seq[j];
        }
    }
}

void JMpcFlt::buildQPProblem(
    const Eigen::VectorXd& current_state,
    const Eigen::VectorXd& last_control,
    const std::vector<Eigen::VectorXd>& reference_path,
    const Eigen::MatrixXd& Psi,
    const Eigen::MatrixXd& Theta,
    Eigen::SparseMatrix<double>& H,
    Eigen::VectorXd& g,
    Eigen::VectorXd& lb,
    Eigen::VectorXd& ub) {
    
    // 构建二次规划问题 (步骤10)
    // 1. 构建H矩阵
    Eigen::MatrixXd Q_bar = kroneckerProduct(
        Eigen::MatrixXd::Identity(Np, Np), Q_);
    Eigen::MatrixXd R_bar = kroneckerProduct(
        Eigen::MatrixXd::Identity(Nc, Nc), R_);
    
    Eigen::MatrixXd H_dense = Theta.transpose() * Q_bar * Theta + R_bar;
    H = H_dense.sparseView();
    
    // 2. 构建g向量
    Eigen::VectorXd xi_0(AUG_STATE_DIM);
    xi_0 << current_state, last_control;
    
    Eigen::VectorXd Y_ref(STATE_DIM * Np);
    for(int i = 0; i < Np; i++) {
        Y_ref.segment(i*STATE_DIM, STATE_DIM) = reference_path[i];
    }
    
    g = Theta.transpose() * Q_bar * (Psi * xi_0 - Y_ref);
    
    // 3. 设置约束
    const int n_du = CONTROL_DIM * Nc;
    lb = Eigen::VectorXd::Constant(n_du, -0.5);  
    ub = Eigen::VectorXd::Constant(n_du, 0.5);   
    
    // 控制量约束
    for(int i = 0; i < Nc; i++) {
        // 速度约束
        double v_min = -0.5;  // 允许小幅后退
        double v_max = 1.0;   // 最大前进速度
        lb(i*CONTROL_DIM) = v_min - last_control(0);
        ub(i*CONTROL_DIM) = v_max - last_control(0);
        
        // 转向角约束
        lb(i*CONTROL_DIM+1) = -max_delta_ - last_control(1);
        ub(i*CONTROL_DIM+1) = max_delta_ - last_control(1);
    }
}

Eigen::VectorXd JMpcFlt::solve(
    const Eigen::VectorXd& current_state,
    const std::vector<Eigen::VectorXd>& reference_path,
    const Eigen::VectorXd& last_control) {
    
    // 计算期望控制序列
    std::vector<Eigen::VectorXd> reference_controls(Np);
    for(int i = 0; i < Np-1; i++) {
        // 计算期望航向角变化率
        double dphi = reference_path[i+1](2) - reference_path[i](2);
        // 根据运动学模型反解控制量
        double v_ref = 0.5;  // 设定一个合理的参考速度
        double delta_ref = atan2(dphi * L_, v_ref * dt_);
        
        Eigen::VectorXd ref_u(CONTROL_DIM);
        ref_u << v_ref, delta_ref;
        reference_controls[i] = ref_u;
    }
    reference_controls[Np-1] = reference_controls[Np-2];
    
    // 线性化和预测
    std::vector<Eigen::MatrixXd> A_tilde_seq(Np);
    std::vector<Eigen::MatrixXd> B_tilde_seq(Np);
    std::vector<Eigen::MatrixXd> C_tilde_seq(Np);
    
    for(int i = 0; i < Np; i++) {
        Eigen::MatrixXd A, B;
        linearizeModel(reference_path[i], reference_controls[i], A, B);
        buildAugmentedSystem(A, B, A_tilde_seq[i], B_tilde_seq[i], C_tilde_seq[i]);
    }
    
    // 2. 构建预测矩阵
    Eigen::MatrixXd Psi, Theta;
    buildPredictionMatrices(A_tilde_seq, B_tilde_seq, C_tilde_seq, Psi, Theta);
    
    // 3. 构建并求解QP问题
    Eigen::SparseMatrix<double> H;
    Eigen::VectorXd g, lb, ub;
    
    buildQPProblem(current_state, last_control, reference_path,
                   Psi, Theta, H, g, lb, ub);
    
    // 4. 配置OSQP求解器
    OSQPSettings* settings = (OSQPSettings*)c_malloc(sizeof(OSQPSettings));
    osqp_set_default_settings(settings);
    settings->alpha = 1.0;
    settings->verbose = false;
    settings->warm_start = true;
    
    // 5. 设置OSQP数据
    OSQPData* data = (OSQPData*)c_malloc(sizeof(OSQPData));
    data->n = H.rows();
    data->m = H.rows();  // 约束数量等于变量数量
    
    // 转换为CSC格式
    Eigen::SparseMatrix<double> H_upper = H.triangularView<Eigen::Upper>();
    std::vector<c_int> H_outer(H_upper.outerIndexPtr(), 
                              H_upper.outerIndexPtr() + H_upper.outerSize() + 1);
    std::vector<c_int> H_inner(H_upper.innerIndexPtr(), 
                              H_upper.innerIndexPtr() + H_upper.nonZeros());
    
    // 创建单位矩阵作为约束矩阵A
    Eigen::SparseMatrix<double> A = Eigen::SparseMatrix<double>(H.rows(), H.rows());
    A.setIdentity();
    std::vector<c_int> A_outer(A.outerIndexPtr(),
                              A.outerIndexPtr() + A.outerSize() + 1);
    std::vector<c_int> A_inner(A.innerIndexPtr(),
                              A.innerIndexPtr() + A.nonZeros());
    
    data->P = csc_matrix(H_upper.rows(), H_upper.cols(),
                        H_upper.nonZeros(),
                        H_upper.valuePtr(),
                        H_inner.data(),
                        H_outer.data());
    
    data->A = csc_matrix(A.rows(), A.cols(),
                        A.nonZeros(),
                        A.valuePtr(),
                        A_inner.data(),
                        A_outer.data());
    
    data->q = g.data();
    data->l = lb.data();
    data->u = ub.data();
    
    // 6. 求解QP问题
    OSQPWorkspace* work;
    osqp_setup(&work, data, settings);
    osqp_solve(work);
    
    // 7. 提取结果
    Eigen::VectorXd delta_u(CONTROL_DIM);
    delta_u << work->solution->x[0], work->solution->x[1];
    
    // 8. 计算实际控制量
    Eigen::VectorXd control = last_control + delta_u;
    
    // 9. 清理内存
    osqp_cleanup(work);
    c_free(data->A);
    c_free(data->P);
    c_free(data);
    c_free(settings);
    
    return control;
} 