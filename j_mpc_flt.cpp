#include "j_mpc_flt.h"
#include <iostream>

JMpcFlt::JMpcFlt() {
    // 初始化MPC参数
    dt_ = 0.1;  // 采样时间100ms
    max_delta_ = 0.7; // 最大转向角 40度
    L_ = 2.7;   // 轴距
    max_v_ = 1.0; // 最大速度
    
    // 初始化权重矩阵
    Q_ = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);
    Q_(0,0) = 100.0;  // x位置误差权重
    Q_(1,1) = 100.0;  // y位置误差权重
    Q_(2,2) = 50.0;   // 航向角误差权重
    
    R_ = Eigen::MatrixXd::Identity(CONTROL_DIM, CONTROL_DIM);
    R_(0,0) = 1.0;    // 速度控制权重
    R_(1,1) = 10.0;   // 转向角控制权重
}

Eigen::MatrixXd JMpcFlt::linearizeModel(
    const Eigen::VectorXd& reference_state,
    const Eigen::VectorXd& reference_control) const {
    
    // 提取参考状态和控制
    double x_ref = reference_state(0);
    double y_ref = reference_state(1);
    double theta_ref = reference_state(2);
    
    double V_ref = reference_control(0);
    double delta_ref = reference_control(1);
    
    // 计算参考点处的状态导数 ẋᵣ
    Eigen::Vector3d x_dot_ref;
    x_dot_ref(0) = V_ref * cos(theta_ref);
    x_dot_ref(1) = V_ref * sin(theta_ref);
    x_dot_ref(2) = V_ref * tan(delta_ref) / L_;
    
    // 1. 计算状态雅可比矩阵 A = ∂f/∂x
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);
    A(0,2) = -V_ref * sin(theta_ref);
    A(1,2) = V_ref * cos(theta_ref);
    
    // 2. 计算控制雅可比矩阵 B = ∂f/∂u
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(STATE_DIM, CONTROL_DIM);
    B(0,0) = cos(theta_ref);
    B(1,0) = sin(theta_ref);
    B(2,0) = tan(delta_ref) / L_;
    B(2,1) = V_ref / (L_ * pow(cos(delta_ref), 2));
    
    // 3. 离散化系统矩阵 Δx(k+1) = (I + dt*A)*Δx(k) + dt*B*Δu(k)
    Eigen::MatrixXd Ad = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM) + dt_ * A;
    Eigen::MatrixXd Bd = dt_ * B;
    
    // 4. 构建增广矩阵 [Ad Bd]
    Eigen::MatrixXd result(STATE_DIM, STATE_DIM + CONTROL_DIM);
    result << Ad, Bd;
    
    return result;
}

Eigen::VectorXd JMpcFlt::solve(
    const Eigen::VectorXd& current_state,
    const std::vector<Eigen::VectorXd>& reference_path) {
    
    // 1. 构建QP问题
    Eigen::SparseMatrix<double> P;  // Hessian矩阵
    Eigen::VectorXd q;              // 线性项
    Eigen::SparseMatrix<double> A;  // 约束矩阵
    Eigen::VectorXd l, u;           // 约束上下界
    
    buildQPProblem(current_state, reference_path, P, q, A, l, u);
    
    // 2. 设置OSQP求解器
    OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
    osqp_set_default_settings(settings);
    settings->alpha = 1.0;  // ADMM步长
    settings->verbose = false;
    
    // 3. 设置OSQP数据
    OSQPData *data = (OSQPData *)c_malloc(sizeof(OSQPData));
    data->n = P.rows();
    data->m = A.rows();
    
    // 转换Eigen矩阵为CSC格式
    Eigen::SparseMatrix<double> P_triu = P.triangularView<Eigen::Upper>();
    std::vector<c_int> P_outer(P_triu.outerIndexPtr(), P_triu.outerIndexPtr() + P_triu.outerSize() + 1);
    std::vector<c_int> P_inner(P_triu.innerIndexPtr(), P_triu.innerIndexPtr() + P_triu.nonZeros());
    data->P = csc_matrix(P_triu.rows(), P_triu.cols(),
                        P_triu.nonZeros(),
                        P_triu.valuePtr(),
                        P_inner.data(),
                        P_outer.data());
    
    data->q = q.data();
    
    std::vector<c_int> A_outer(A.outerIndexPtr(), A.outerIndexPtr() + A.outerSize() + 1);
    std::vector<c_int> A_inner(A.innerIndexPtr(), A.innerIndexPtr() + A.nonZeros());
    data->A = csc_matrix(A.rows(), A.cols(),
                        A.nonZeros(),
                        A.valuePtr(),
                        A_inner.data(),
                        A_outer.data());
    
    data->l = l.data();
    data->u = u.data();
    
    // 4. 初始化求解器
    OSQPWorkspace *work;
    osqp_setup(&work, data, settings);
    
    // 5. 求解QP问题
    osqp_solve(work);
    
    // 6. 提取结果
    Eigen::VectorXd result(CONTROL_DIM);
    result(0) = work->solution->x[STATE_DIM * (HORIZON + 1)];      // 速度
    result(1) = work->solution->x[STATE_DIM * (HORIZON + 1) + 1];  // 转向角
    
    // 7. 清理
    osqp_cleanup(work);
    c_free(data->A);
    c_free(data->P);
    c_free(data);
    c_free(settings);
    
    return result;
}

void JMpcFlt::buildQPProblem(
    const Eigen::VectorXd& current_state,
    const std::vector<Eigen::VectorXd>& reference_path,
    Eigen::SparseMatrix<double>& P,
    Eigen::VectorXd& q,
    Eigen::SparseMatrix<double>& A,
    Eigen::VectorXd& l,
    Eigen::VectorXd& u) {
    
    const int n_vars = STATE_DIM * (HORIZON + 1) + CONTROL_DIM * HORIZON;
    const int n_cons = STATE_DIM * (HORIZON + 1);
    
    // 初始化矩阵
    P.resize(n_vars, n_vars);
    q.resize(n_vars);
    A.resize(n_cons, n_vars);
    l.resize(n_cons);
    u.resize(n_cons);
    
    std::vector<Eigen::Triplet<double>> P_triplets;
    std::vector<Eigen::Triplet<double>> A_triplets;
    
    // 构建目标函数
    for (int i = 0; i < HORIZON; i++) {
        // 状态代价
        for (int j = 0; j < STATE_DIM; j++) {
            P_triplets.push_back(Eigen::Triplet<double>(
                STATE_DIM * i + j,
                STATE_DIM * i + j,
                Q_(j,j)));
            q(STATE_DIM * i + j) = -Q_(j,j) * reference_path[i][j];
        }
        
        // 控制代价
        for (int j = 0; j < CONTROL_DIM; j++) {
            P_triplets.push_back(Eigen::Triplet<double>(
                STATE_DIM * (HORIZON + 1) + CONTROL_DIM * i + j,
                STATE_DIM * (HORIZON + 1) + CONTROL_DIM * i + j,
                R_(j,j)));
        }
    }
    
    // 构建约束条件
    // 初始状态约束
    for (int j = 0; j < STATE_DIM; j++) {
        A_triplets.push_back(Eigen::Triplet<double>(j, j, 1.0));
        l(j) = current_state(j);
        u(j) = current_state(j);
    }
    
    // 创建参考控制序列（简单起见，设为零）
    std::vector<Eigen::VectorXd> reference_control(HORIZON);
    for(int i = 0; i < HORIZON; i++) {
        reference_control[i] = Eigen::VectorXd::Zero(CONTROL_DIM);
    }
    
    // 动力学约束
    for (int i = 0; i < HORIZON; i++) {
        Eigen::MatrixXd AB = linearizeModel(reference_path[i], reference_control[i]);
        Eigen::MatrixXd Ad = AB.leftCols(STATE_DIM);
        Eigen::MatrixXd Bd = AB.rightCols(CONTROL_DIM);
        
        for (int j = 0; j < STATE_DIM; j++) {
            // Δx(k+1) = (I + dt*A)*Δx(k) + dt*B*Δu(k)
            A_triplets.push_back(Eigen::Triplet<double>(
                STATE_DIM * (i + 1) + j,
                STATE_DIM * (i + 1) + j,
                -1.0));
                
            for (int k = 0; k < STATE_DIM; k++) {
                A_triplets.push_back(Eigen::Triplet<double>(
                    STATE_DIM * (i + 1) + j,
                    STATE_DIM * i + k,
                    Ad(j,k)));
            }
            
            for (int k = 0; k < CONTROL_DIM; k++) {
                A_triplets.push_back(Eigen::Triplet<double>(
                    STATE_DIM * (i + 1) + j,
                    STATE_DIM * (HORIZON + 1) + CONTROL_DIM * i + k,
                    Bd(j,k)));
            }
            
            l(STATE_DIM * (i + 1) + j) = 0;
            u(STATE_DIM * (i + 1) + j) = 0;
        }
    }
    
    P.setFromTriplets(P_triplets.begin(), P_triplets.end());
    A.setFromTriplets(A_triplets.begin(), A_triplets.end());
}

 