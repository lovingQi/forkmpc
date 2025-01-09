#include "j_mpc_flt.h" 
#include <iostream>

JMpcFlt::JMpcFlt() {
    // 初始化系统参数
    dt_ = 0.1;
    max_delta_ = 0.7;
    L_ = 2.7;
    max_v_ = 1.0;
    rho_ = 1e3;
    
    // 调整权重矩阵
    Q_ = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);
    Q_(0,0) = 10.0;       // x位置误差
    Q_(1,1) = 1200.0;     // 进一步增大y位置误差权重
    Q_(2,2) = 1000.0;     // 进一步增大航向角误差权重
    
    R_ = Eigen::MatrixXd::Identity(CONTROL_DIM, CONTROL_DIM);
    R_(0,0) = 0.5;        // 速度增量权重
    R_(1,1) = 0.01;       // 显著降低转向角增量权重
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
    
    // 动态分配矩阵大小
    Psi = Eigen::MatrixXd::Zero(STATE_DIM * Np, AUG_STATE_DIM);
    Theta = Eigen::MatrixXd::Zero(STATE_DIM * Np, CONTROL_DIM * Nc);
    
    // 构建Psi矩阵
    Eigen::MatrixXd temp = Eigen::MatrixXd::Identity(AUG_STATE_DIM, AUG_STATE_DIM);
    Psi.block(0, 0, STATE_DIM, AUG_STATE_DIM) = C_tilde_seq[0] * temp;
    
    for(int i = 1; i < Np; i++) {
        temp = A_tilde_seq[i-1] * temp;
        Psi.block(i*STATE_DIM, 0, STATE_DIM, AUG_STATE_DIM) = C_tilde_seq[i] * temp;
    }
    
    // 构建Theta矩阵
    for(int i = 0; i < Np; i++) {
        for(int j = 0; j < std::min(i+1, Nc); j++) {
            if (i == j) {
                Theta.block(i*STATE_DIM, j*CONTROL_DIM, STATE_DIM, CONTROL_DIM) = 
                    C_tilde_seq[i] * B_tilde_seq[j];
            } else {
                Eigen::MatrixXd temp = B_tilde_seq[j];
                for(int k = j+1; k <= i; k++) {
                    temp = A_tilde_seq[k-1] * temp;
                }
                Theta.block(i*STATE_DIM, j*CONTROL_DIM, STATE_DIM, CONTROL_DIM) = 
                    C_tilde_seq[i] * temp;
            }
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
    
    // 动态构建权重矩阵
    Eigen::MatrixXd Q_bar = kroneckerProduct(
        Eigen::MatrixXd::Identity(Np, Np), Q_);
    Eigen::MatrixXd R_bar = kroneckerProduct(
        Eigen::MatrixXd::Identity(Nc, Nc), R_);
    
    // 确保H矩阵维度正确
    Eigen::MatrixXd H_dense = Theta.transpose() * Q_bar * Theta + R_bar;
    H = H_dense.sparseView();
    
    // 动态构建参考轨迹向量
    Eigen::VectorXd Y_ref(STATE_DIM * Np);
    for(int i = 0; i < Np; i++) {
        Y_ref.segment(i*STATE_DIM, STATE_DIM) = reference_path[i];
    }
    
    // 2. 构建g向量
    Eigen::VectorXd xi_0(AUG_STATE_DIM);
    xi_0 << current_state, last_control;
    
    g = Theta.transpose() * Q_bar * (Psi * xi_0 - Y_ref);
    
    // 3. 设置约束
    const int n_du = CONTROL_DIM * Nc;
    lb = Eigen::VectorXd::Constant(n_du, -0.5);  
    ub = Eigen::VectorXd::Constant(n_du, 0.5);   
    
    // 根据横向偏差和航向角误差调整速度约束
    double lateral_error = std::abs(current_state(1));
    double heading_error = std::abs(std::atan2(std::sin(current_state(2)), std::cos(current_state(2))));
    
    // 在误差大时更激进地降低速度上限
    double v_max = 0.8 * std::exp(-1.0 * lateral_error) * std::exp(-1.2 * heading_error);
    v_max = std::max(0.15, v_max);  // 进一步降低最小速度
    
    // 控制量约束
    for(int i = 0; i < Nc; i++) {
        // 速度约束
        double v_min = -0.15;  // 允许更小的后退速度
        lb(i*CONTROL_DIM) = v_min - last_control(0);
        ub(i*CONTROL_DIM) = v_max - last_control(0);
        
        // 转向角约束（更激进的转向策略）
        double delta_range = max_delta_ * (1.0 - 0.1 * std::exp(-0.1 * lateral_error));
        lb(i*CONTROL_DIM+1) = -delta_range - last_control(1);
        ub(i*CONTROL_DIM+1) = delta_range - last_control(1);
    }
}

void JMpcFlt::adjustPredictionHorizon(double lateral_error, double curvature) {
    // 基础预测步长
    int base_Np = 20;
    
    // 根据横向误差和曲率调整预测步长
    int horizon_adjustment = static_cast<int>(5.0 * std::abs(lateral_error) + 
                                            10.0 * std::abs(curvature));
    
    // 设置新的预测和控制步长（减小调整幅度）
    Np = std::min(30, base_Np + horizon_adjustment);  // 最大不超过30步
    Nc = std::min(25, Np - 5);  // 控制步长略小于预测步长
}

Eigen::VectorXd JMpcFlt::solve(
    const Eigen::VectorXd& current_state,
    const std::vector<Eigen::VectorXd>& reference_path,
    const Eigen::VectorXd& last_control) {
    
    // 首先计算曲率和调整预测步长
    double curvature = 0;
    if (reference_path.size() > 2) {
        int mid = std::min(20, (int)reference_path.size()/2);
        double dx = reference_path[mid](0) - reference_path[0](0);
        double dy = reference_path[mid](1) - reference_path[0](1);
        double dphi = reference_path[mid](2) - reference_path[0](2);
        double ds = std::sqrt(dx*dx + dy*dy);
        curvature = std::abs(dphi) / (ds + 1e-6);
    }
    
    // 调整预测步长
    double lateral_error = std::abs(current_state(1));
    adjustPredictionHorizon(lateral_error, curvature);
    
    // 然后创建本地路径副本并确保长度足够
    std::vector<Eigen::VectorXd> local_ref_path = reference_path;
    while (local_ref_path.size() < Np) {
        local_ref_path.push_back(reference_path.back());
    }
    
    // 计算期望控制序列
    std::vector<Eigen::VectorXd> reference_controls(Np);
    for(int i = 0; i < Np-1; i++) {
        // 根据曲率调整参考速度
        double dx = local_ref_path[i+1](0) - local_ref_path[i](0);
        double dy = local_ref_path[i+1](1) - local_ref_path[i](1);
        double dphi = local_ref_path[i+1](2) - local_ref_path[i](2);
        double ds = std::sqrt(dx*dx + dy*dy);
        double curvature = std::abs(dphi) / (ds + 1e-6);
        
        // 速度随曲率增大而降低
        double v_ref = 0.8 * std::exp(-2.0 * curvature);
        v_ref = std::max(0.2, v_ref);
        
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
        linearizeModel(local_ref_path[i], reference_controls[i], A, B);
        buildAugmentedSystem(A, B, A_tilde_seq[i], B_tilde_seq[i], C_tilde_seq[i]);
    }
    
    // 2. 构建预测矩阵
    Eigen::MatrixXd Psi, Theta;
    buildPredictionMatrices(A_tilde_seq, B_tilde_seq, C_tilde_seq, Psi, Theta);
    
    // 3. 构建并求解QP问题
    Eigen::SparseMatrix<double> H;
    Eigen::VectorXd g, lb, ub;
    
    buildQPProblem(current_state, last_control, local_ref_path,
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