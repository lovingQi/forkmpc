#pragma once

#include "Eigen/Dense"
#include <vector>
#include <memory>
#include "osqp/osqp.h"

class MPCController {
public:
    MPCController(
        double dt,          // 采样时间
        double L,           // 轴距
        int pred_horizon,   // 预测时域
        double max_steer,   // 最大转向角
        double max_speed    // 最大速度
    );
    
    ~MPCController();
    
    // 禁用拷贝
    MPCController(const MPCController&) = delete;
    MPCController& operator=(const MPCController&) = delete;
    
    bool solve(
        const Eigen::Vector3d& current_state,
        const std::vector<Eigen::Vector2d>& ref_path,
        double& steer,
        double& speed
    );

private:
    // 系统参数
    const double dt_;
    const double L_;
    const int pred_horizon_;
    const double max_steer_;
    const double max_speed_;
    
    // OSQP数据结构
    struct OSQPDeleter {
        void operator()(OSQPData* ptr) {
            if (ptr) {
                if (ptr->P) {
                    c_free(ptr->P->x);
                    c_free(ptr->P->i);
                    c_free(ptr->P->p);
                    c_free(ptr->P);
                }
                if (ptr->A) {
                    c_free(ptr->A->x);
                    c_free(ptr->A->i);
                    c_free(ptr->A->p);
                    c_free(ptr->A);
                }
                if (ptr->q) c_free(ptr->q);
                if (ptr->l) c_free(ptr->l);
                if (ptr->u) c_free(ptr->u);
                c_free(ptr);
            }
        }
    };
    
    struct OSQPWorkspaceDeleter {
        void operator()(OSQPWorkspace* ptr) {
            if (ptr) osqp_cleanup(ptr);
        }
    };
    
    std::unique_ptr<OSQPData, OSQPDeleter> data_;
    std::unique_ptr<OSQPWorkspace, OSQPWorkspaceDeleter> workspace_;
    
    // 权重矩阵
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd R_;
    
    // 辅助函数
    void linearizeModel(
        const Eigen::Vector3d& state,
        const Eigen::Vector2d& input,
        Eigen::MatrixXd& A,
        Eigen::MatrixXd& B
    );
    
    void setupQPProblem(
        const Eigen::Vector3d& current_state,
        const std::vector<Eigen::Vector2d>& ref_path
    );
}; 