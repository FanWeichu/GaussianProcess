#include "gaussian_process.h"


// std::cout << << std::endl;

//GaussianProcess::GaussianProcess() {
//  gp_data_set_
//}
void GaussianProcess::setScaleParameter(Eigen::VectorXd scale_parameter) {
  scale_parameter_ = scale_parameter;
}
void GaussianProcess::setDimension(int m, int n) {
  m_ = m;
  n_ = n;
}

void GaussianProcess::LoadCSVData() {
  std::ifstream data("../model_data.csv");
  std::string line;
  std::vector<std::string> model_data_raw;
  std::vector<double> model_data_vec;
  while (std::getline(data, line)) {
    model_data_raw.push_back(line);
  }
  for (int i = 0; i < model_data_raw.size(); ++i) {
    std::replace(model_data_raw[i].begin(), model_data_raw[i].end(), '\r', ' ');
    std::istringstream ssline(model_data_raw[i]);
    std::string tmp;
    while (std::getline(ssline, tmp, ',')) {
      model_data_vec.push_back(std::atof(tmp.c_str()));
    }
  }
  int n_line = model_data_raw.size();
  int n_state = model_data_vec.size() / n_line;
  int n_used = 6;
  Eigen::MatrixXd model_data_raw_matrix;

//  x y vx vy ax ay beta r dr
  model_data_raw_matrix = Eigen::MatrixXd::Zero(n_line, n_state);

  int tmp_cnt = 0;
  for (int i = 0; i < n_line; ++i) {
    for (int j = 0; j < n_state; ++j) {
      model_data_raw_matrix.block(i, j, 1, 1) << model_data_vec[tmp_cnt];
      tmp_cnt++;
    }
  }

//  select which states should be observed
  model_data_used_matrix_ = Eigen::MatrixXd::Zero(n_line, n_used);
//  gp_data_set_ = Eigen::MatrixXd::Zero(999, n_used);

  model_data_used_matrix_.block(0, 0, n_line, 1) = model_data_raw_matrix.block(0, 4, n_line, 1);
  model_data_used_matrix_.block(0, 1, n_line, 1) = model_data_raw_matrix.block(0, 5, n_line, 1);
  model_data_used_matrix_.block(0, 2, n_line, 1) = model_data_raw_matrix.block(0, 2, n_line, 1);
  model_data_used_matrix_.block(0, 3, n_line, 1) = model_data_raw_matrix.block(0, 3, n_line, 1);
  model_data_used_matrix_.block(0, 4, n_line, 1) = model_data_raw_matrix.block(0, 7, n_line, 1);
  model_data_used_matrix_.block(0, 5, n_line, 1) = model_data_raw_matrix.block(0, 8, n_line, 1);

  for (int i = 0; i < n_used; ++i) {
    model_data_used_matrix_.col(i) = model_data_used_matrix_.col(i) * scale_parameter_[i];
  }
//  for (int i = 0; i < 40; ++i) {
//    std::cout << model_data_used_matrix_.block(i, 0, 1, n_used) << std::endl;
//  }
  int a = 1;
}

void GaussianProcess::dataPreprocess(Eigen::MatrixXd measure_data, const double K_diff) {
//  std::cout << "-----------------------------------------" << std::endl;
//  std::cout << "measured data : "<<measure_data<< std::endl;

  int m_data_set = gp_data_set_.rows();
  if (m_data_set == 0) {
    gp_data_set_ = Eigen::MatrixXd::Zero(1, m_);
//    gp_data_set_.row(0) << measure_data.row(0);
//    std::cout << gp_data_set_ << std::endl;
  }
  m_data_set = gp_data_set_.rows();
//  Eigen::MatrixXd condi_vec(m_data_set, 1);

  bool flag;
  double condi = 0.;
  for (int i = 0; i < m_data_set; ++i) {
    std::vector<double> condi_vec;
    Eigen::VectorXd err = gp_data_set_.row(i) - measure_data;
//    std::cout <<"err : "<< err.dot(err) << std::endl;
    condi_vec.push_back(err.dot(err));
    condi = *std::min_element(condi_vec.begin(), condi_vec.end());
//    std::cout <<"condi : "<< condi << std::endl;
  }

  if (condi > K_diff) {
    flag = true;
    gp_data_set_.conservativeResize(gp_data_set_.rows() + 1,
                                    gp_data_set_.cols());
    gp_data_set_.row(gp_data_set_.rows() - 1) = measure_data;
//    std::cout << "gp_data_set_ : "<<gp_data_set_<< std::endl;
  } else {
    flag = false;
  }
  int a = 1;
}
void GaussianProcess::genCovarianceMatrix() {
  int m_cov = gp_data_set_.rows();
  
  covariance_matrix_ = Eigen::MatrixXd::Zero(m_cov, m_cov);
  for (int i = 0; i < m_cov; ++i) {
    for (int j = 0; j < m_cov; ++j) {
      Eigen::VectorXd diff_row = gp_data_set_.row(i) - gp_data_set_.row(j);
      double l2_norm = diff_row.dot(diff_row);
      covariance_matrix_.block(i,j,1,1) << exp(-l2_norm / 2.);
    }
  }
}
double GaussianProcess::calMuPrediction(Eigen::MatrixXd x_measure,
                                        Eigen::MatrixXd data_set,
                                        Eigen::MatrixXd cov_matrix,
                                        int prediction_state) {
  Eigen::MatrixXd K_xy;
  K_xy = Eigen::MatrixXd::Zero(cov_matrix.rows(), 1);
  for (int i = 0; i < cov_matrix.rows(); ++i) {
    Eigen::VectorXd diff_row = gp_data_set_.row(i) - x_measure;
    double l2_norm = diff_row.dot(diff_row);
    K_xy.row(i) << exp(-l2_norm / 2.);
  }

//  double mu;

  std::cout << "x_measure : " << x_measure << std::endl;
//  std::cout << "data_set.col(prediction_state) : " << data_set.col(prediction_state) << std::endl;
//  std::cout << "data_set : " << data_set << std::endl;

//  std::cout << "K_xy :" << K_xy << std::endl;
  std::cout << "cov_matrix :" << cov_matrix << std::endl;
  std::cout << "part 1 : " << (K_xy.transpose() * cov_matrix.inverse()) << std::endl;

//  std::cout << "mu : " << (K_xy.transpose() * cov_matrix.inverse()) * data_set.col(prediction_state)<< std::endl;

  auto tmp = (K_xy.transpose() * cov_matrix.inverse()) * data_set.col(prediction_state);
  double mu = tmp.value();
//  mu = 1.;


  return mu;
}
