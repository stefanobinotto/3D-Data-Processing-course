#include "Registration.h"


struct PointDistance
{ 
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // This class should include an auto-differentiable cost function. 
  // To rotate a point given an axis-angle rotation, use
  // the Ceres function:
  // AngleAxisRotatePoint(...) (see ceres/rotation.h)
  // Similarly to the Bundle Adjustment case initialize the struct variables with the source and  the target point.
  // You have to optimize only the 6-dimensional array (rx, ry, rz, tx ,ty, tz).
  // WARNING: When dealing with the AutoDiffCostFunction template parameters,
  // pay attention to the order of the template parameters
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  Eigen::Vector3d target,source;

  PointDistance(Eigen::Vector3d target, Eigen::Vector3d source) : target(target), source(source) {}

  //Compute the residual
  template <typename T>
  bool operator()(const T* const transformation, T* residual) const {
    
    //store the source cloud
    Eigen::Matrix<T,3,1> data{{T(source[0])}, {T(source[1])}, {T(source[2])}};
    
    //apply rotation to source cloud
    Eigen::Matrix<T,3,1> transformed_source;
    ceres::AngleAxisRotatePoint(transformation, data.data(), transformed_source.data());

    //apply translation to source cloud
    transformed_source += Eigen::Matrix<T,3,1>(transformation[3], transformation[4], transformation[5]);

    //The error is the difference between the target and transformed_source
    residual[0] = transformed_source[0] - T(target[0]);
    residual[1] = transformed_source[1] - T(target[1]);
    residual[2] = transformed_source[2] - T(target[2]);

    return true;
  }

  // Create cost function.
  static ceres::CostFunction* Create(const Eigen::Vector3d target,
                                     const Eigen::Vector3d source) {
    return (new ceres::AutoDiffCostFunction<PointDistance, 3, 6>(new PointDistance(target, source)));
  }  
};


Registration::Registration(std::string cloud_source_filename, std::string cloud_target_filename)
{
  open3d::io::ReadPointCloud(cloud_source_filename, source_ );
  open3d::io::ReadPointCloud(cloud_target_filename, target_ );
  Eigen::Vector3d gray_color;
  source_for_icp_ = source_;
}


Registration::Registration(open3d::geometry::PointCloud cloud_source, open3d::geometry::PointCloud cloud_target)
{
  source_ = cloud_source;
  target_ = cloud_target;
  source_for_icp_ = source_;
}


void Registration::draw_registration_result()
{
  //clone input
  open3d::geometry::PointCloud source_clone = source_;
  open3d::geometry::PointCloud target_clone = target_;

  //different color
  Eigen::Vector3d color_s;
  Eigen::Vector3d color_t;
  color_s<<1, 0.706, 0;
  color_t<<0, 0.651, 0.929;

  target_clone.PaintUniformColor(color_t);
  source_clone.PaintUniformColor(color_s);
  source_clone.Transform(transformation_);

  auto src_pointer =  std::make_shared<open3d::geometry::PointCloud>(source_clone);
  auto target_pointer =  std::make_shared<open3d::geometry::PointCloud>(target_clone);
  open3d::visualization::DrawGeometries({src_pointer, target_pointer});
  return;
}



void Registration::execute_icp_registration(double threshold, int max_iteration, double relative_rmse, std::string mode)
{ 
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //ICP main loop
  //Check convergence criteria and the current iteration.
  //If mode=="svd" use get_svd_icp_transformation if mode=="lm" use get_lm_icp_transformation.
  //Remember to update transformation_ class variable, you can use source_for_icp_ to store transformed 3d points.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  //initialize previous rmse measurement
  double prev_rmse = std::numeric_limits<double>::infinity();
  
  for (int i = 0; i<max_iteration; i++) {

    //get target-source correspondences and rmse
    std::tuple<std::vector<size_t>, std::vector<size_t>, double> correspondences = find_closest_point(threshold);

    //Convergency check
    //First check: the rmse didn't improve
    //Second check: the rmse didn't improve enough
    if ((prev_rmse < std::get<2>(correspondences)) || (std::abs(std::get<2>(correspondences) - prev_rmse) <= relative_rmse)) {
      break; //exit the for cycle
    }

    //update prev_rmse
    prev_rmse = std::get<2>(correspondences);

    //store transformation matrix, depending on lm or svd approaches
    Eigen::Matrix4d registration_matrix;
    if(mode == "svd") {
      registration_matrix = get_svd_icp_transformation(std::get<0>(correspondences), std::get<1>(correspondences));
    } else if (mode == "lm"){
      registration_matrix = get_lm_icp_registration(std::get<0>(correspondences), std::get<1>(correspondences));
    }

    //update transformation_
    set_transformation(transformation_*registration_matrix);

    std::cout<<"RMSE of iteration "<<i<<"/"<<max_iteration<<" = "<<prev_rmse<<std::endl;
  }

  return;
}


std::tuple<std::vector<size_t>, std::vector<size_t>, double> Registration::find_closest_point(double threshold)
{ ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Find source and target indices: for each source point find the closest one in the target and discard if their 
  //distance is bigger than threshold
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  std::vector<size_t> target_indices;
  std::vector<size_t> source_indices;
  double mse;

  //Apply transformation_ to the source cloud
  open3d::geometry::PointCloud source_clone = source_;
  source_clone.Transform(transformation_);

  //Find number of points
  int num_source_points  = source_clone.points_.size();

  //Initialize KD-Tree for efficient K-NN computation
  open3d::geometry::KDTreeFlann target_kd_tree(target_);

  //Index of the target matched point
  std::vector<int> idx(1);

  //Distance between source and target matched points
  std::vector<double> dist2(1);

  for (size_t i = 0; i < num_source_points; i++) {
    //store the current source point
    Eigen::Vector3d source_point = source_clone.points_[i];
    //find the target nearest neighbor of the source point
    target_kd_tree.SearchKNN(source_point, 1, idx, dist2);

    //accept the correspondence if the distance is smaller than the threshold
    if (dist2[0]<=threshold) {
      source_indices.push_back(i);
      target_indices.push_back(idx[0]);
      mse = mse * i/(i+1) + dist2[0]/(i+1);
    }
  }

  double rmse = sqrt(mse);

  return {source_indices, target_indices, rmse};
}

Eigen::Matrix4d Registration::get_svd_icp_transformation(std::vector<size_t> source_indices, std::vector<size_t> target_indices){
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Find point clouds centroids and subtract them. 
  //Use SVD (Eigen::JacobiSVD<Eigen::MatrixXd>) to find best rotation and translation matrix.
  //Use source_indices and target_indices to extract point to compute the 3x3 matrix to be decomposed.
  //Remember to manage the special reflection case.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity(4,4);
  
  //Apply transformation_ to the current source cloud
  open3d::geometry::PointCloud source_clone = source_;
  source_clone.Transform(transformation_);

  //Computation of centroids
  Eigen::Vector3d source_centroid(0,0,0), target_centroid(0,0,0);

  for (int i = 0; i < source_clone.points_.size(); i++) {
    source_centroid.x() += source_clone.points_[i].x();
    source_centroid.y() += source_clone.points_[i].y();
    source_centroid.z() += source_clone.points_[i].z();
  }
  for (int i = 0; i < target_.points_.size(); i++) {
    target_centroid.x() += target_.points_[i].x();
    target_centroid.y() += target_.points_[i].y();
    target_centroid.z() += target_.points_[i].z();
  }

  source_centroid/=((double)source_clone.points_.size());
  target_centroid/=((double)target_.points_.size());

  //Computation of W matrix
  Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
  for (int i = 0; i < source_indices.size(); i++ ) {
    Eigen::Vector3d source_point = source_clone.points_[source_indices[i]];
    Eigen::Vector3d target_point = target_.points_[target_indices[i]];

    W += (target_point-target_centroid)*(source_point-source_centroid).transpose();
  }
  
  //Compute SVD decomposition of W, get U and V matrices only
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();

  //Compute rotation matrix
  Eigen::Matrix3d R = U*V.transpose();
  
  //in case of reflection, negate the last column of V
  if (R.determinant() == -1) {
    Eigen::DiagonalMatrix<double,3> diag(1,1,-1);
    R = U*diag*V.transpose();
  }

  //Compute translation matrix
  Eigen::Vector3d t = target_centroid - R*source_centroid;

  //Aggregate the transformations
  transformation.block<3,3>(0,0) = R;
  transformation.block<3,1>(0,3) = t;

  return transformation;
}

Eigen::Matrix4d Registration::get_lm_icp_registration(std::vector<size_t> source_indices, std::vector<size_t> target_indices)
{
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Use LM (Ceres) to find best rotation and translation matrix. 
  //Remember to convert the euler angles in a rotation matrix, store it coupled with the final translation on:
  //Eigen::Matrix4d transformation.
  //The first three elements of std::vector<double> transformation_arr represent the euler angles, the last ones
  //the translation.
  //use source_indices and target_indices to extract point to compute the matrix to be decomposed.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity(4,4);

  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = 4;
  options.max_num_iterations = 100;

  //Apply transformation_ to the current source cloud
  open3d::geometry::PointCloud source_clone = source_;
  source_clone.Transform(transformation_);

  ceres::Problem problem;
  std::vector<double> transformation_arr(6, 0.0);
  int num_points = source_indices.size();

  // For each point....
  for( int i = 0; i < num_points; i++ ) {
    
    //Save target and source matched points
    Eigen::Vector3d source_point = source_clone.points_[source_indices[i]];
    Eigen::Vector3d target_point = target_.points_[target_indices[i]];

    //Declare cost function and residual
    ceres::CostFunction* cost_function = PointDistance::Create(target_point, source_point);
    problem.AddResidualBlock(cost_function, nullptr, transformation_arr.data());
  }

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  //Convertion of the euler angles into the rotation matrix using Roll, Pitch and Yaw
  Eigen::AngleAxisd Roll(transformation_arr[0], Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd Pitch(transformation_arr[1], Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd Yaw(transformation_arr[2], Eigen::Vector3d::UnitZ());
  Eigen::Matrix3d R;
  R = Roll * Pitch * Yaw;

  //Store translation vector
  Eigen::Vector3d t(transformation_arr[3], transformation_arr[4], transformation_arr[5]);

  //Aggregate the transformations
  transformation.block<3,3>(0,0) = R;
  transformation.block<3,1>(0,3) = t;

  return transformation;
}


void Registration::set_transformation(Eigen::Matrix4d init_transformation)
{
  transformation_=init_transformation;
}


Eigen::Matrix4d  Registration::get_transformation()
{
  return transformation_;
}

double Registration::compute_rmse()
{
  open3d::geometry::KDTreeFlann target_kd_tree(target_);
  open3d::geometry::PointCloud source_clone = source_;
  source_clone.Transform(transformation_);
  int num_source_points  = source_clone.points_.size();
  Eigen::Vector3d source_point;
  std::vector<int> idx(1);
  std::vector<double> dist2(1);
  double mse;
  for(size_t i=0; i < num_source_points; ++i) {
    source_point = source_clone.points_[i];
    target_kd_tree.SearchKNN(source_point, 1, idx, dist2);
    mse = mse * i/(i+1) + dist2[0]/(i+1);
  }
  return sqrt(mse);
}

void Registration::write_tranformation_matrix(std::string filename)
{
  std::ofstream outfile (filename);
  if (outfile.is_open())
  {
    outfile << transformation_;
    outfile.close();
  }
}

void Registration::save_merged_cloud(std::string filename)
{
  //clone input
  open3d::geometry::PointCloud source_clone = source_;
  open3d::geometry::PointCloud target_clone = target_;

  source_clone.Transform(transformation_);
  open3d::geometry::PointCloud merged = target_clone+source_clone;
  open3d::io::WritePointCloud(filename, merged );
}