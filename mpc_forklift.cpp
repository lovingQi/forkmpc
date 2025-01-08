#include "mpc_forklift.h"
#include <cmath>
#include <algorithm>
#include <iostream>

MPCController::MPCController(double dt, double L, int pred_horizon, 
                           double max_steer, double max_speed)
    : dt_(dt), L_(L), pred_horizon_(pred_horizon),
      max_steer_(max_steer), max_speed_(max_speed),
      data_(static_cast<OSQPData*>(c_malloc(sizeof(OSQPData)))) {
    
    // 初始化权重矩阵
    Q_ = Eigen::MatrixXd::Identity(3, 3);
    Q_(0,0) = 10.0;    // 降低位置权重，使运动更平滑
    Q_(1,1) = 10.0;
    Q_(2,2) = 50.0;    // 保持适度的航向角权重
    
    R_ = Eigen::MatrixXd::Identity(2, 2);
    R_(0,0) = 1.0;    // 增加转向权重，防止剧烈转向
    R_(1,1) = 0.1;    // 保持较小的速度权重
    
    // 初始化OSQP数据
    data_->n = 0;
    data_->m = 0;
    data_->P = nullptr;
    data_->A = nullptr;
    data_->q = nullptr;
    data_->l = nullptr;
    data_->u = nullptr;

    // 初始化一个空的QP问题
    Eigen::Vector3d init_state = Eigen::Vector3d::Zero();
    std::vector<Eigen::Vector2d> init_path;
    init_path.push_back(Eigen::Vector2d::Zero());
    setupQPProblem(init_state, init_path);
}

void MPCController::linearizeModel(const Eigen::Vector3d& state,
                                 const Eigen::Vector2d& input,
                                 Eigen::MatrixXd& A,
                                 Eigen::MatrixXd& B) {
    double theta = state(2);
    double delta = input(0);
    double v = input(1);
    
    // 计算雅可比矩阵 A
    A = Eigen::MatrixXd::Identity(3, 3);
    A(0,2) = -v * sin(theta) * cos(delta) * dt_;
    A(1,2) = v * cos(theta) * cos(delta) * dt_;
    
    // 计算雅可比矩阵 B
    B = Eigen::MatrixXd::Zero(3, 2);
    B(0,0) = -v * cos(theta) * sin(delta) * dt_;
    B(0,1) = cos(theta) * cos(delta) * dt_;
    B(1,0) = -v * sin(theta) * sin(delta) * dt_;
    B(1,1) = sin(theta) * cos(delta) * dt_;
    B(2,0) = v * cos(delta) / L_ * dt_;
    B(2,1) = sin(delta) / L_ * dt_;
}

bool MPCController::solve(const Eigen::Vector3d& current_state,
                         const std::vector<Eigen::Vector2d>& ref_path,
                         double& steer, double& speed) {
    setupQPProblem(current_state, ref_path);
    
    if (workspace_ == nullptr) {
        std::cout << "Error: OSQP setup failed!" << std::endl;
        return false;
    }
    
    // 求解QP问题
    osqp_solve(workspace_.get());
    
    // 检查求解状态
    if (workspace_.get()->info->status_val != 1) {
        std::cout << "OSQP solve failed with status: " 
                  << workspace_.get()->info->status_val << std::endl;
        return false;
    }
    
    // 提取第一个控制输入
    steer = workspace_.get()->solution->x[0];
    speed = workspace_.get()->solution->x[1];
    
    return true;
}

void MPCController::setupQPProblem(const Eigen::Vector3d& current_state,
                                 const std::vector<Eigen::Vector2d>& ref_path) {
    // 如果存在旧的工作空间，先清理
    if (workspace_ != nullptr) {
        workspace_.reset(nullptr);
    }
    
    // 清理旧的内存
    if (data_.get()->P) {
        c_free(data_.get()->P->x);
        c_free(data_.get()->P->i);
        c_free(data_.get()->P->p);
        c_free(data_.get()->P);
    }
    if (data_.get()->A) {
        c_free(data_.get()->A->x);
        c_free(data_.get()->A->i);
        c_free(data_.get()->A->p);
        c_free(data_.get()->A);
    }
    if (data_.get()->q) c_free(data_.get()->q);
    if (data_.get()->l) c_free(data_.get()->l);
    if (data_.get()->u) c_free(data_.get()->u);
    
    int nx = 3;  // 状态维度
    int nu = 2;  // 控制维度
    int N = pred_horizon_;
    
    int n_variables = nx * N + nu * N;  // 状态和控制变量
    int n_constraints = nx * (N + 1);   // 动力学约束和控制约束
    
    // 初始化矩阵维度
    data_.get()->n = n_variables;
    data_.get()->m = n_constraints;
    
    std::cout << "Problem dimensions: " << std::endl;
    std::cout << "n_variables: " << n_variables << std::endl;
    std::cout << "n_constraints: " << n_constraints << std::endl;
    
    // 获取线性化系统矩阵
    Eigen::Vector2d nominal_input(0, 0);
    Eigen::MatrixXd Ad, Bd;
    linearizeModel(current_state, nominal_input, Ad, Bd);
    std::cout << "Ad matrix: \n" << Ad << std::endl;
    std::cout << "Bd matrix: \n" << Bd << std::endl;
    
    // 构建目标函数矩阵 P
    Eigen::MatrixXd P_eigen = Eigen::MatrixXd::Zero(n_variables, n_variables);
    for (int i = 0; i < N; i++) {
        // 状态代价
        P_eigen.block(i*nx, i*nx, nx, nx) = Q_;
        // 控制代价
        P_eigen.block(N*nx + i*nu, N*nx + i*nu, nu, nu) = R_;
    }
    
    // 确保P是对称的并且正定
    P_eigen = (P_eigen + P_eigen.transpose()) * 0.5;
    // 增加对角线项以确保正定性
    P_eigen.diagonal().array() += 1e-3;
    
    // 构建约束矩阵 A
    Eigen::MatrixXd A_eigen = Eigen::MatrixXd::Zero(n_constraints, n_variables);
    
    // 设置动力学约束
    for (int i = 0; i < N; i++) {
        // 当前状态
        A_eigen.block(i*nx, i*nx, nx, nx) = Ad;
        // 下一个状态
        if (i < N-1) {
            A_eigen.block(i*nx, (i+1)*nx, nx, nx) = -Eigen::MatrixXd::Identity(nx, nx);
        }
        // 控制输入影响
        A_eigen.block(i*nx, N*nx + i*nu, nx, nu) = Bd;
    }
    
    // 打印A矩阵的一部分用于调试
    std::cout << "A matrix block (0,0): \n" << A_eigen.block(0, 0, nx, nx) << std::endl;
    std::cout << "A matrix block (0,N*nx): \n" << A_eigen.block(0, N*nx, nx, nu) << std::endl;
    
    // 设置约束上下界
    Eigen::VectorXd l_eigen = Eigen::VectorXd::Zero(n_constraints);
    Eigen::VectorXd u_eigen = Eigen::VectorXd::Zero(n_constraints);
    
    // 设置约束
    for (int i = 0; i < N; i++) {
        // 状态约束
        if (i == 0) {
            l_eigen.segment(i*nx, nx) = current_state;
            u_eigen.segment(i*nx, nx) = current_state;
        } else {
            // 使用更合理的状态约束范围
            l_eigen.segment(i*nx, nx) << -10.0, -10.0, -M_PI;  // x, y, theta
            u_eigen.segment(i*nx, nx) << 10.0, 10.0, M_PI;
        }
    }
    
    // 构建线性项 q
    Eigen::VectorXd q_eigen = Eigen::VectorXd::Zero(n_variables);
    for (int i = 0; i < N; i++) {
        if (i < ref_path.size()) {
            double dx = ref_path[i].x() - current_state[0];
            double dy = ref_path[i].y() - current_state[1];
            double desired_theta = atan2(dy, dx);
            double current_theta = current_state[2];
            
            double theta_error = desired_theta - current_theta;
            while (theta_error > M_PI) theta_error -= 2*M_PI;
            while (theta_error < -M_PI) theta_error += 2*M_PI;
            
            double distance = std::sqrt(dx*dx + dy*dy);
            
            // 参考Apollo的目标设置
            double target_speed = std::min(max_speed_, std::max(0.5, 2.0 * distance));
            // 状态目标
            q_eigen(i*nx) = -10.0 * dx;      // 增加x位置目标权重
            q_eigen(i*nx + 1) = -10.0 * dy;  // 增加y位置目标权重
            q_eigen(i*nx + 2) = -50.0 * theta_error;  // 增加航向角目标权重
            // 控制目标
            q_eigen(N*nx + i*nu) = 0;  // 转向目标保持为0
            q_eigen(N*nx + i*nu + 1) = -5.0 * target_speed;  // 增加速度目标权重
        }
    }
    
    // 转换P矩阵为CSC格式
    std::vector<c_float> P_x;
    std::vector<c_int> P_i;
    std::vector<c_int> P_p;
    P_p.push_back(0);
    
    for(int col = 0; col < n_variables; col++) {
        for(int row = 0; row <= col; row++) {
            if(std::abs(P_eigen(row,col)) > 1e-9) {
                P_x.push_back(P_eigen(row,col));
                P_i.push_back(row);
            }
        }
        P_p.push_back(P_i.size());
    }
    std::cout << "P matrix non-zeros: " << P_x.size() << std::endl;
    
    // 确保内存分配正确
    if (P_x.empty() || P_i.empty() || P_p.empty()) {
        std::cout << "Error: Empty P matrix!" << std::endl;
        return;
    }

    // 转换A矩阵为CSC格式
    std::vector<c_float> A_x;
    std::vector<c_int> A_i;
    std::vector<c_int> A_p;
    A_p.push_back(0);
    
    for(int col = 0; col < n_variables; col++) {
        for(int row = 0; row < n_constraints; row++) {
            if(std::abs(A_eigen(row,col)) > 1e-9) {
                A_x.push_back(A_eigen(row,col));
                A_i.push_back(row);
            }
        }
        A_p.push_back(A_i.size());
    }
    
    // 确保内存分配正确
    if (A_x.empty() || A_i.empty() || A_p.empty()) {
        std::cout << "Error: Empty A matrix!" << std::endl;
        return;
    }

    // 分配OSQP数据内存
    if (n_variables <= 0 || n_constraints <= 0) {
        std::cout << "Error: Invalid problem dimensions!" << std::endl;
        return;
    }
    c_float* q = (c_float*)c_malloc(n_variables * sizeof(c_float));
    c_float* l = (c_float*)c_malloc(n_constraints * sizeof(c_float));
    c_float* u = (c_float*)c_malloc(n_constraints * sizeof(c_float));
    
    if (!q || !l || !u) {
        std::cout << "Error: Memory allocation failed!" << std::endl;
        return;
    }

    // 复制向量数据
    for(int i = 0; i < n_variables; i++) {
        q[i] = q_eigen(i);
    }
    for(int i = 0; i < n_constraints; i++) {
        l[i] = l_eigen(i);
        u[i] = u_eigen(i);
    }

    // 设置OSQP数据
    std::cout << "Setting up OSQP data..." << std::endl;
    c_float* P_x_ptr = (c_float*)c_malloc(P_x.size() * sizeof(c_float));
    c_int* P_i_ptr = (c_int*)c_malloc(P_i.size() * sizeof(c_int));
    c_int* P_p_ptr = (c_int*)c_malloc(P_p.size() * sizeof(c_int));
    std::cout << "Memory allocated for P matrix" << std::endl;
    
    memcpy(P_x_ptr, P_x.data(), P_x.size() * sizeof(c_float));
    memcpy(P_i_ptr, P_i.data(), P_i.size() * sizeof(c_int));
    memcpy(P_p_ptr, P_p.data(), P_p.size() * sizeof(c_int));
    std::cout << "Data copied for P matrix" << std::endl;
    data_.get()->P = csc_matrix(n_variables, n_variables, P_x.size(), P_x_ptr, P_i_ptr, P_p_ptr);

    c_float* A_x_ptr = (c_float*)c_malloc(A_x.size() * sizeof(c_float));
    c_int* A_i_ptr = (c_int*)c_malloc(A_i.size() * sizeof(c_int));
    c_int* A_p_ptr = (c_int*)c_malloc(A_p.size() * sizeof(c_int));
    memcpy(A_x_ptr, A_x.data(), A_x.size() * sizeof(c_float));
    memcpy(A_i_ptr, A_i.data(), A_i.size() * sizeof(c_int));
    memcpy(A_p_ptr, A_p.data(), A_p.size() * sizeof(c_int));
    data_.get()->A = csc_matrix(n_constraints, n_variables, A_x.size(), A_x_ptr, A_i_ptr, A_p_ptr);

    data_.get()->q = q;
    data_.get()->l = l;
    data_.get()->u = u;

    // 设置求解器
    OSQPSettings* settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
    osqp_set_default_settings(settings);

    OSQPWorkspace* workspace_ptr = nullptr;
    osqp_setup(&workspace_ptr, data_.get(), settings);
    workspace_.reset(workspace_ptr);

    c_free(settings);
}

// 在析构函数中清理
MPCController::~MPCController() {
    // No manual cleanup needed
} 