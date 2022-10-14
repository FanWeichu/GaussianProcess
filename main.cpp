#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <functional>
#include <Eigen/Dense>
#include <typeinfo>
#include "gaussian_process.h"

int main() {

  std::string data_analyse_folder = "../data_analyse";
  std::ofstream OutFile;
  if (std::filesystem::exists(data_analyse_folder)) {
    auto csv_path = data_analyse_folder.append("/GP_result.csv");
    OutFile.open(csv_path, std::ios::out);
  } else {
    std::filesystem::create_directories(data_analyse_folder);
    auto csv_path = data_analyse_folder.append("/GP_result.csv");
    OutFile.open(csv_path, std::ios::out);
  }


  int m = 6;
  int n = 2;

  //  important, too little threshold leads to Divergence
  double gp_threshold = 0.1;
  GaussianProcess gp;
  Eigen::VectorXd scale_parameter(m);
  scale_parameter << 1/0.5, 1., 1./4., 1., 1./0.6, 1./5;
  gp.setDimension(m, n);
  gp.setScaleParameter(scale_parameter);
  gp.LoadCSVData();
  
//  simulation
  for (int i = 0; i < gp.model_data_used_matrix_.rows(); ++i) {
    gp.dataPreprocess(gp.model_data_used_matrix_.row(i), gp_threshold);
  }
  gp.genCovarianceMatrix();

  int csv_title = 0;

  for (int i = 0; i < gp.gp_data_set_.rows(); ++i) {
    double mu;
    int prediction_state = 4;
    mu = gp.calMuPrediction(gp.gp_data_set_.row(i), gp.gp_data_set_,
                            gp.covariance_matrix_, prediction_state);

    //    store for data_plot
    if (csv_title < 1) {
      OutFile << "t" << ",";
      OutFile << "mu" << ",";
      OutFile << "prediction_state" << ","
      << std::endl;
      csv_title++;
    } else {
      OutFile << i + 1 << ",";
      OutFile << mu << ",";
      OutFile << gp.gp_data_set_.block(i,prediction_state,1,1) << ","
      << std::endl;
    }
//    std::cout << "mu_final : " << mu << std::endl;
  }



//  std::cout << gp.covariance_matrix_ << std::endl;
//  std::cout << gp.covariance_matrix_.rows() << std::endl;


  return 0;
}
