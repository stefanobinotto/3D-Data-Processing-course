#ifndef MVS_REG_REGISTRATION_H
#define MVS_REG_REGISTRATION_H
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include "open3d/Open3D.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>

class Registration {

public:

  //Initialize the source and tharget point clouds to be processed
  Registration(std::string cloud_source_filename, std::string cloud_target_filename);
  Registration(open3d::geometry::PointCloud cloud_source, open3d::geometry::PointCloud cloud_target);

  //Visualize source and target point clouds
  void draw_registration_result();

  void execute_icp_registration(double threshold = 0.02, int max_iteration = 100, double relative_rmse = 1e-6, std::string mode="svd");
  void set_transformation(Eigen::Matrix4d init_transformation);

  //Get the current transformation matrix needed to align the source to the target cloud
  Eigen::Matrix4d get_transformation();

  void write_tranformation_matrix(std::string filename);
  void save_merged_cloud(std::string filename);

  //Compute the RMSE between the points of the source and target clouds
  double compute_rmse();

private:

  //For each point in the source point cloud find the closest one in the target
  std::tuple<std::vector<size_t>, std::vector<size_t>,  double> find_closest_point(double threshold);

  Eigen::Matrix4d get_svd_icp_transformation(std::vector<size_t> source_indices, std::vector<size_t> target_indices);
  Eigen::Matrix4d get_lm_icp_registration(std::vector<size_t> source_indices, std::vector<size_t> target_indices);
  open3d::geometry::PointCloud source_;
  open3d::geometry::PointCloud source_for_icp_;
  open3d::geometry::PointCloud target_;
  Eigen::Matrix4d transformation_ = Eigen::Matrix4d::Identity();


};


#endif //MVS_REG_REGISTRATION_H
