
#ifndef GP_MPC__GAUSSIAN_PROCESS_H_
#define GP_MPC__GAUSSIAN_PROCESS_H_

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

class GaussianProcess {
 public:
//  GaussianProcess();
//  virtual ~GaussianProcess();
  void LoadCSVData();
  void setScaleParameter(Eigen::VectorXd scale_parameter);
  void setDimension(int m, int n);
  void dataPreprocess(Eigen::MatrixXd measure_data, const double K_diff);
  void genCovarianceMatrix();
  double calMuPrediction(Eigen::MatrixXd x_measure, Eigen::MatrixXd data_set,
                         Eigen::MatrixXd cov_matrix, int prediction_state);



  Eigen::VectorXd scale_parameter_;
  Eigen::MatrixXd model_data_used_matrix_;
  Eigen::MatrixXd gp_data_set_;
  Eigen::MatrixXd covariance_matrix_;

//  observer states
  int m_;
//  prediction states
  int n_;
};

#endif //GP_MPC__GAUSSIAN_PROCESS_H_
