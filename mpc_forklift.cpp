#include "mpc_forklift.h"
#include <cmath>
#include <algorithm>

MPCController::MPCController(double dt, double L, int pred_horizon, 
                           double max_steer, double max_speed)
    : dt_(dt), L_(L), pred_horizon_(pred_horizon),
      max_steer_(max_steer), max_speed_(max_speed),
      data_(static_cast<OSQPData*>(c_malloc(sizeof(OSQPData)))) {
    
    // 初始化权重矩阵
    Q_ = Eigen::MatrixXd::Identity(3, 3);
    Q_(0,0) = 100.0;
    Q_(1,1) = 100.0;
    Q_(2,2) = 10.0;
    
    R_ = Eigen::MatrixXd::Identity(2, 2);
    R_(0,0) = 1.0;
    R_(1,1) = 1.0;
    
    // 初始化OSQP数据
    data_->n = 0;
    data_->m = 0;
    data_->P = nullptr;
    data_->A = nullptr;
    data_->q = nullptr;
    data_->l = nullptr;
    data_->u = nullptr;
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
    
    // 求解QP问题
    osqp_solve(workspace_.get());
    
    // 检查求解状态
    if (workspace_.get()->info->status_val != 1) {
        return false;
    }
    
    // 提取第一个控制输入
    steer = workspace_.get()->solution->x[0];
    speed = workspace_.get()->solution->x[1];
    
    return true;
}

void MPCController::setupQPProblem(const Eigen::Vector3d& current_state,
                                 const std::vector<Eigen::Vector2d>& ref_path) {
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
    
    int n_variables = nu * N;  // 只优化控制输入
    int n_constraints = nu * N;  // 控制约束
    
    // 初始化矩阵
    data_.get()->n = n_variables;
    data_.get()->m = n_constraints;
    
    // 构建目标函数矩阵 P
    Eigen::MatrixXd P_eigen = Eigen::MatrixXd::Zero(n_variables, n_variables);
    for (int i = 0; i < N; i++) {
        P_eigen.block(i*nu, i*nu, nu, nu) = R_;
    }
    
    // 确保P是对称的
    P_eigen = (P_eigen + P_eigen.transpose()) * 0.5;
    
    // 构建线性项 q
    Eigen::VectorXd q_eigen = Eigen::VectorXd::Zero(n_variables);
    for (int i = 0; i < N; i++) {
        // 参考轨迹代价
        if (i < ref_path.size()) {  // 防止越界
            q_eigen.segment(i*nu, 2) = -2 * Q_.block(0,0,2,2) * 
                Eigen::Vector2d(ref_path[i].x(), ref_path[i].y());
        }
    }
    
    // 构建约束矩阵 A（包含系统动力学）
    Eigen::MatrixXd A_eigen = Eigen::MatrixXd::Identity(n_constraints, n_variables);
    
    // 设置约束上下界
    Eigen::VectorXd l_eigen = Eigen::VectorXd::Zero(n_constraints);
    Eigen::VectorXd u_eigen = Eigen::VectorXd::Zero(n_constraints);
    
    // 设置控制量约束
    for (int i = 0; i < N; i++) {
        // 转向角约束
        l_eigen(i*nu) = -max_steer_;
        u_eigen(i*nu) = max_steer_;
        // 速度约束
        l_eigen(i*nu + 1) = 0;  // 速度下限
        u_eigen(i*nu + 1) = max_speed_;
    }

    Eigen::Vector2d nominal_input(0, 0);  // 用于线性化的标称输入
    Eigen::MatrixXd Ad, Bd;
    linearizeModel(current_state, nominal_input, Ad, Bd);
    
    // 转换P矩阵为CSC格式
    std::vector<c_float> P_x;  // 非零元素
    std::vector<c_int> P_i;    // 行索引
    std::vector<c_int> P_p;    // 列指针
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

    // 分配OSQP数据内存
    c_float* q = (c_float*)c_malloc(n_variables * sizeof(c_float));
    c_float* l = (c_float*)c_malloc(n_constraints * sizeof(c_float));
    c_float* u = (c_float*)c_malloc(n_constraints * sizeof(c_float));

    // 复制向量数据
    for(int i = 0; i < n_variables; i++) {
        q[i] = q_eigen(i);
    }
    for(int i = 0; i < n_constraints; i++) {
        l[i] = l_eigen(i);
        u[i] = u_eigen(i);
    }

    // 设置OSQP数据
    c_float* P_x_ptr = (c_float*)c_malloc(P_x.size() * sizeof(c_float));
    c_int* P_i_ptr = (c_int*)c_malloc(P_i.size() * sizeof(c_int));
    c_int* P_p_ptr = (c_int*)c_malloc(P_p.size() * sizeof(c_int));
    memcpy(P_x_ptr, P_x.data(), P_x.size() * sizeof(c_float));
    memcpy(P_i_ptr, P_i.data(), P_i.size() * sizeof(c_int));
    memcpy(P_p_ptr, P_p.data(), P_p.size() * sizeof(c_int));
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

    if (workspace_ != nullptr) {
        osqp_cleanup(workspace_.get());
    }
    OSQPWorkspace* workspace_ptr = nullptr;
    osqp_setup(&workspace_ptr, data_.get(), settings);
    workspace_.reset(workspace_ptr);

    c_free(settings);
}

// 在析构函数中清理
MPCController::~MPCController() {
    // No manual cleanup needed
} 