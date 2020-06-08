// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2015 Pierre Moulon.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "openMVG/sfm/sfm_data_BA_ceres.hpp"

#ifdef OPENMVG_USE_OPENMP
#include <omp.h>
#endif

#include "ceres/problem.h"
#include "ceres/solver.h"
#include "openMVG/cameras/Camera_Common.hpp"
#include "openMVG/cameras/Camera_Intrinsics.hpp"
#include "openMVG/geometry/Similarity3.hpp"
#include "openMVG/geometry/Similarity3_Kernel.hpp"
//- Robust estimation - LMeds (since no threshold can be defined)
#include "openMVG/robust_estimation/robust_estimator_LMeds.hpp"
#include "openMVG/sfm/sfm_data_BA_ceres_camera_functor.hpp"
#include "openMVG/sfm/sfm_data_transform.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/types.hpp"

#include <ceres/rotation.h>
#include <ceres/types.h>

#include <iostream>
#include <limits>
#include <boost/filesystem.hpp>

namespace openMVG {
namespace sfm {

using namespace openMVG::cameras;
using namespace openMVG::geometry;

// Ceres CostFunctor used for SfM pose center to GPS pose center minimization
//The main idea is to make use of planes and minimize the distance between 2 planes (LIDAR + SfM)
struct LineReprojectionConstraintCostFunction
{
  const double * m_line_2d_endpoints; //4D vector
  explicit LineReprojectionConstraintCostFunction(const double* const line_2d)
  :m_line_2d_endpoints(line_2d)
  {
  }
  template <typename T> bool operator() 
  (
  const T* const cam_intrinsics,
  const T* const cam_extrinsics,
  const T* const line_3d_endpoint,
  T* out_residuals
  ) 
  const 
  {
    const T * cam_R = cam_extrinsics;
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> cam_t(&cam_extrinsics[3]);

    Eigen::Matrix<T, 3, 1> transformed_point_start, transformed_point_end;
    // Rotate the point according the camera rotation
    ceres::AngleAxisRotatePoint(cam_R, line_3d_endpoint, transformed_point_start.data()); //3D starting point
    ceres::AngleAxisRotatePoint(cam_R, line_3d_endpoint+3, transformed_point_end.data()); //3D end point

    // Apply the camera translation
    transformed_point_start += cam_t;
    transformed_point_end += cam_t;

    // Transform the point from homogeneous to euclidean (undistorted point)
    Eigen::Matrix<T, 2, 1> projected_point3d_start = transformed_point_start.hnormalized();
    Eigen::Matrix<T, 2, 1> projected_point3d_end = transformed_point_end.hnormalized();

    //--
    // Apply intrinsic parameters
    //--

    const T& focal = cam_intrinsics[0];
    const T& principal_point_x = cam_intrinsics[1];
    const T& principal_point_y = cam_intrinsics[2];
    // Apply focal length and principal point to get the final image coordinates
    Eigen::Matrix<T,2,1> proj_3d_point_start(principal_point_x + projected_point3d_start.x() * focal,
                                                   principal_point_y + projected_point3d_start.y() * focal);

    Eigen::Matrix<T,2,1> proj_3d_point_end(principal_point_x + projected_point3d_end.x() * focal,
                                           principal_point_y + projected_point3d_end.y() * focal);

    //Compute orthogonal projection of 2D point on a 2D line
    const Eigen::Matrix<T,2,1> line_2d_start(m_line_2d_endpoints[0], m_line_2d_endpoints[1]);
    const Eigen::Matrix<T,2,1> line_2d_end(m_line_2d_endpoints[2], m_line_2d_endpoints[3]);
    
    
    //We project on m_line_2d_endpoint_start and end on the (finite) line formed by (proj_3d_point_start, proj_3d_point_end)
    Eigen::Matrix<T,2,1> start_2d_proj_3d_line ;
    Eigen::Matrix<T,2,1> end_2d_proj_3d_line;
    
    //Step 1: Compute the projection 
    Eigen::Matrix<T,2,1> proj_3d_dir = end_2d_proj_3d_line - start_2d_proj_3d_line;
    proj_3d_dir.normalize();

    const T t_start = std::min(T(1), std::max(T(0), T(proj_3d_dir.dot(line_2d_start - proj_3d_point_start))));
    const T t_end = std::min(T(1), std::max(T(0), T(proj_3d_dir.dot(line_2d_end - proj_3d_point_start))));

    start_2d_proj_3d_line = proj_3d_point_start + t_start*proj_3d_dir;
    end_2d_proj_3d_line = proj_3d_point_start + t_end*proj_3d_dir;

    //Distance between 2d endpoint projected on the projected 3D line and the 2d endpoint
    T dist_start_to_start = pow(pow(start_2d_proj_3d_line.x() - line_2d_start.x(), 2.)+pow(start_2d_proj_3d_line.y() - line_2d_start.y(), 2.), 1/2.);
    T dist_end_to_end = pow(pow(end_2d_proj_3d_line.x() - line_2d_end.x(), 2.) + pow(end_2d_proj_3d_line.y() - line_2d_end.y(), 2.), 1/2.);

    // Compute and return the error is the difference between the predicted
    //  and observed position
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(out_residuals);

    residuals <<  dist_start_to_start, 
                  dist_end_to_end;

      return true;
  }
};
struct PoseCenterConstraintCostFunction
{
  Vec3 weight_;
  Vec3 pose_center_constraint_;

  PoseCenterConstraintCostFunction
  (
    const Vec3 & center,
    const Vec3 & weight
  ): weight_(weight), pose_center_constraint_(center)
  {
  }

  template <typename T> bool
  operator()
  (
    const T* const cam_extrinsics, // R_t
    T* residuals
  )
  const
  {
    using Vec3T = Eigen::Matrix<T,3,1>;
    Eigen::Map<const Vec3T> cam_R(&cam_extrinsics[0]);
    Eigen::Map<const Vec3T> cam_t(&cam_extrinsics[3]);
    const Vec3T cam_R_transpose(-cam_R);

    Vec3T pose_center;
    // Rotate the point according the camera rotation
    ceres::AngleAxisRotatePoint(cam_R_transpose.data(), cam_t.data(), pose_center.data());
    pose_center = pose_center * T(-1);

    Eigen::Map<Vec3T> residuals_eigen(residuals);
    residuals_eigen = weight_.cast<T>().cwiseProduct(pose_center - pose_center_constraint_.cast<T>());

    return true;
  }
};

/// Create the appropriate cost functor according the provided input camera intrinsic model.
/// The residual can be weighetd if desired (default 0.0 means no weight).
ceres::CostFunction * IntrinsicsToCostFunction
(
  IntrinsicBase * intrinsic,
  const Vec2 & observation,
  const double weight
)
{
  switch (intrinsic->getType())
  {
    case PINHOLE_CAMERA:
      return ResidualErrorFunctor_Pinhole_Intrinsic::Create(observation, weight);
    case PINHOLE_CAMERA_RADIAL1:
      return ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K1::Create(observation, weight);
    case PINHOLE_CAMERA_RADIAL3:
      return ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K3::Create(observation, weight);
    case PINHOLE_CAMERA_BROWN:
      return ResidualErrorFunctor_Pinhole_Intrinsic_Brown_T2::Create(observation, weight);
    case PINHOLE_CAMERA_FISHEYE:
      return ResidualErrorFunctor_Pinhole_Intrinsic_Fisheye::Create(observation, weight);
    case CAMERA_SPHERICAL:
      return ResidualErrorFunctor_Intrinsic_Spherical::Create(intrinsic, observation, weight);
    default:
      return {};
  }
}

Bundle_Adjustment_Ceres::BA_Ceres_options::BA_Ceres_options
(
  const bool bVerbose,
  bool bmultithreaded
)
: bVerbose_(bVerbose),
  nb_threads_(1),
  parameter_tolerance_(1e-8), //~= numeric_limits<float>::epsilon()
  bUse_loss_function_(true)
{
  #ifdef OPENMVG_USE_OPENMP
    nb_threads_ = omp_get_max_threads();
  #endif // OPENMVG_USE_OPENMP
  if (!bmultithreaded)
    nb_threads_ = 1;

  bCeres_summary_ = false;

  // Default configuration use a DENSE representation
  linear_solver_type_ = ceres::DENSE_SCHUR;
  preconditioner_type_ = ceres::JACOBI;
  // If Sparse linear solver are available
  // Descending priority order by efficiency (SUITE_SPARSE > CX_SPARSE > EIGEN_SPARSE)
  if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::SUITE_SPARSE))
  {
    sparse_linear_algebra_library_type_ = ceres::SUITE_SPARSE;
    linear_solver_type_ = ceres::SPARSE_SCHUR;
  }
  else
  {
    if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::CX_SPARSE))
    {
      sparse_linear_algebra_library_type_ = ceres::CX_SPARSE;
      linear_solver_type_ = ceres::SPARSE_SCHUR;
    }
    else
    if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::EIGEN_SPARSE))
    {
      sparse_linear_algebra_library_type_ = ceres::EIGEN_SPARSE;
      linear_solver_type_ = ceres::SPARSE_SCHUR;
    }
  }
}


Bundle_Adjustment_Ceres::Bundle_Adjustment_Ceres
(
  const Bundle_Adjustment_Ceres::BA_Ceres_options & options
)
: ceres_options_(options)
{}

Bundle_Adjustment_Ceres::BA_Ceres_options &
Bundle_Adjustment_Ceres::ceres_options()
{
  return ceres_options_;
}

bool Bundle_Adjustment_Ceres::Adjust
(
  SfM_Data & sfm_data,     // the SfM scene to refine
  const Optimize_Options & options
)
{
  //----------
  // Add camera parameters
  // - intrinsics
  // - poses [R|t]

  // Create residuals for each observation in the bundle adjustment problem. The
  // parameters for cameras and points are added automatically.
  //----------


  double pose_center_robust_fitting_error = 0.0;
  openMVG::geometry::Similarity3 sim_to_center;
  bool b_usable_prior = false;
  if (options.use_motion_priors_opt && sfm_data.GetViews().size() > 3)
  {
    // - Compute a robust X-Y affine transformation & apply it
    // - This early transformation enhance the conditionning (solution closer to the Prior coordinate system)
    {
      // Collect corresponding camera centers
      std::vector<Vec3> X_SfM, X_GPS;
      for (const auto & view_it : sfm_data.GetViewsPriors())
      {
        ViewPriors *prior(view_it.second.get());


        if (prior != nullptr && prior->b_use_pose_center_ && sfm_data.IsPoseAndIntrinsicDefined(prior))
        {
          X_SfM.push_back( sfm_data.GetPoses().at(prior->id_pose).center() );
          X_GPS.push_back( prior->pose_center_ );
        }
      }
      std::cout << " Size X_SfM: " << X_SfM.size() << std::endl;
      std::cout << " Size X_GPS: " << X_GPS.size() << std::endl; 
      openMVG::geometry::Similarity3 sim;

      // Compute the registration:
      if (X_GPS.size() > 3)
      {
        const Mat X_SfM_Mat = Eigen::Map<Mat>(X_SfM[0].data(),3, X_SfM.size());
        const Mat X_GPS_Mat = Eigen::Map<Mat>(X_GPS[0].data(),3, X_GPS.size());
        geometry::kernel::Similarity3_Kernel kernel(X_SfM_Mat, X_GPS_Mat);
        const double lmeds_median = openMVG::robust::LeastMedianOfSquares(kernel, &sim);
        if (lmeds_median != std::numeric_limits<double>::max())
        {
          b_usable_prior = true; // PRIOR can be used safely

          // Compute the median residual error once the registration is applied
          for (Vec3 & pos : X_SfM) // Transform SfM poses for residual computation
          {
            pos = sim(pos);
          }
          Vec residual = (Eigen::Map<Mat3X>(X_SfM[0].data(), 3, X_SfM.size()) - Eigen::Map<Mat3X>(X_GPS[0].data(), 3, X_GPS.size())).colwise().norm();
          std::sort(residual.data(), residual.data() + residual.size());
          pose_center_robust_fitting_error = residual(residual.size()/2);

          // Apply the found transformation to the SfM Data Scene
          openMVG::sfm::ApplySimilarity(sim, sfm_data);

          // Move entire scene to center for better numerical stability
          Vec3 pose_centroid = Vec3::Zero();
          for (const auto & pose_it : sfm_data.poses)
          {
            pose_centroid += (pose_it.second.center() / (double)sfm_data.poses.size());
          }
          sim_to_center = openMVG::geometry::Similarity3(openMVG::sfm::Pose3(Mat3::Identity(), pose_centroid), 1.0);
          openMVG::sfm::ApplySimilarity(sim_to_center, sfm_data, true);
        }
      }
    }
  }
  
  Hash_Map<IndexT, Eigen::Vector6d> all_3d_lines;
  Hash_Map<IndexT, std::pair<IndexT, Eigen::Vector4d>> all_2d_lines;

  std::vector<std::pair<IndexT, std::vector<IndexT>>> potential_matches; //For each line L_j the vector l_x^{xj}

  if (options.use_lines_opt){
    std::cout << " Using lines opt" << std::endl;
    //Right now we are precomputing lines (i.e. fusing lidar clouds + getting SfM points on line)
     std::vector<std::string> allFilenames;
    Eigen::Matrix4d lidar2camera;
    lidar2camera << -0.70992163, -0.02460003, -0.70385092, -0.04874569,
                     0.70414167, -0.00493623, -0.71004236, -0.05289342,  
                     0.01399269, -0.99968519,  0.02082624, -0.04699275,
                        0.        ,  0.        ,  0.,  1.;

    IndexT curIdxLine(0);
    for(auto view_it:sfm_data.views){
        const View* v = view_it.second.get();
        std::string vPath = v->s_Img_path;
        size_t pos = vPath.find("png");
        std::string lPath = vPath.replace(pos, 3,"pcd");

        allFilenames.push_back(lPath);
        boost::filesystem::path curP;
        curP /= sfm_data.s_root_path;
        curP /= "/"+vPath;

        std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> res;
        openMVG::sfm::getLinesInImageAfm(curP.string(), res);
        for(auto elem: res){
          all_2d_lines.insert({curIdxLine, std::make_pair(v->id_view, Eigen::Vector4d(elem.first(0), elem.first(1), elem.second(0), elem.second(1)))});
          ++curIdxLine;
        }
    }

    // Step 1: Load all lines from files

    // Step 2: Load all 3D lines from files 

    // Step 3: Load all 3D clouds
    //SFM cloud
    PointCloudXYZ::Ptr sfmCloud(new PointCloudXYZ);

    for(auto & structure_landmark_it : sfm_data.structure){
        sfmCloud->push_back(pcl::PointXYZ(structure_landmark_it.second.X(0), structure_landmark_it.second.X(1), structure_landmark_it.second.X(2)));
    }
    //Lidar cloud
    PointCloudXYZ::Ptr lidarCloud(new PointCloudXYZ); 
    std::vector<PointCloudXYZ::Ptr> allLidarClouds = openMVG::sfm::readAllClouds(sfm_data.s_root_path, allFilenames);
    openMVG::sfm::fusePointClouds(allLidarClouds,sfm_data.poses, lidar2camera, lidarCloud);
    std::cerr << "Fused point clouds " << std::endl;
    openMVG::sfm::visualizePointCloud(lidarCloud);


    std::vector<std::vector<IndexT>> lines_2d_per_image;
    std::vector<std::vector<IndexT>> lines_3d_per_image;

    //Get all potential matches

    for (const auto& view_it: sfm_data.GetViews()){

      const Pose3 pose = sfm_data.GetPoseOrDie(view_it.second.get());
      const Mat3 K = dynamic_cast<cameras::Pinhole_Intrinsic *>(sfm_data.intrinsics.at(view_it.second->id_intrinsic).get())->K();

      std::vector<std::pair<IndexT, Eigen::Vector4d>> projected_3d_lines = getAllVisibleLines(all_3d_lines, pose, K, view_it.second.get());

      for(auto id_2d_segment: lines_2d_per_image.at(view_it.second->id_view)){
        Eigen::Vector4d cur_2d_segment = all_2d_lines.at(id_2d_segment).second;
        int res = getLineLineCorrespondence(cur_2d_segment, projected_3d_lines, K, pose);
        if (res > 0)
          potential_matches.at(res).second.push_back(id_2d_segment);
      }
    }

   //Remove all lines matched in only one image
   for(auto it = potential_matches.begin();it!=potential_matches.end();++it){
     IndexT curImg = all_2d_lines.at(it->second.at(0)).first; //check that this is the image
     bool valid = false;

     for (auto iI = 1;iI<it->second.size();++iI)
        if (all_2d_lines.at(it->second.at(iI)).first != curImg)
          valid = true;

     if (!valid){
       it = potential_matches.erase(it);
     }else{
       ++it;
     }
   }

    // std::vector<Eigen::Vector6f> lines;

    // std::vector<pcl::ModelCoefficients> planes;
    // std::vector<PointCloudXYZ::Ptr, Eigen::aligned_allocator<PointCloudXYZ::Ptr>> outputClouds;
    // openMVG::sfm::extractPlanesFromCloud(lidarCloud, planes, outputClouds);
    // std::cerr << "Extracted planes..." << std::endl;
    // openMVG::sfm::findLinesFromPlanes(planes, lines);
    // std::cerr << "Found plane intersections..." << std::endl;
    // openMVG::sfm::checkInlierLines(lines, sfmCloud, lineFeatureCorrespondences);

    // for(auto & structure_landmark_it : sfm_data.structure)
        // landmarksLine.push_back(-1);

    // int iLine(0);
    // for(auto elem: *lineFeatureCorrespondences){
        // all3dLines.push_back(elem.first);
        // for(auto landmark: elem.second){
          // landmarksLine.at(landmark) = static_cast<IndexT>(iLine); 
        // }
        // ++iLine;
    // }
  }

  ceres::Problem problem;

  // Data wrapper for refinement:
  Hash_Map<IndexT, std::vector<double>> map_intrinsics;
  Hash_Map<IndexT, std::vector<double>> map_poses;
  Hash_Map<IndexT, std::vector<double>> map_lines;

  // Setup Poses data & subparametrization
  for (const auto & pose_it : sfm_data.poses)
  {
    const IndexT indexPose = pose_it.first;

    const Pose3 & pose = pose_it.second;
    const Mat3 R = pose.rotation();
    const Vec3 t = pose.translation();

    double angleAxis[3];
    ceres::RotationMatrixToAngleAxis((const double*)R.data(), angleAxis);
    // angleAxis + translation
    map_poses[indexPose] = {angleAxis[0], angleAxis[1], angleAxis[2], t(0), t(1), t(2)};

    double * parameter_block = &map_poses.at(indexPose)[0];
    problem.AddParameterBlock(parameter_block, 6);
    if (options.extrinsics_opt == Extrinsic_Parameter_Type::NONE)
    {
      // set the whole parameter block as constant for best performance
      problem.SetParameterBlockConstant(parameter_block);
    }
    else  // Subset parametrization
    {
      std::vector<int> vec_constant_extrinsic;
      // If we adjust only the translation, we must set ROTATION as constant
      if (options.extrinsics_opt == Extrinsic_Parameter_Type::ADJUST_TRANSLATION)
      {
        // Subset rotation parametrization
        vec_constant_extrinsic.insert(vec_constant_extrinsic.end(), {0,1,2});
      }
      // If we adjust only the rotation, we must set TRANSLATION as constant
      if (options.extrinsics_opt == Extrinsic_Parameter_Type::ADJUST_ROTATION)
      {
        // Subset translation parametrization
        vec_constant_extrinsic.insert(vec_constant_extrinsic.end(), {3,4,5});
      }
      if (!vec_constant_extrinsic.empty())
      {
        ceres::SubsetParameterization *subset_parameterization =
          new ceres::SubsetParameterization(6, vec_constant_extrinsic);
        problem.SetParameterization(parameter_block, subset_parameterization);
      }
    }
  }

  // Setup Intrinsics data & subparametrization
  for (const auto & intrinsic_it : sfm_data.intrinsics)
  {
    const IndexT indexCam = intrinsic_it.first;

    if (isValid(intrinsic_it.second->getType()))
    {
      map_intrinsics[indexCam] = intrinsic_it.second->getParams();
      if (!map_intrinsics.at(indexCam).empty())
      {
        double * parameter_block = &map_intrinsics.at(indexCam)[0];
        problem.AddParameterBlock(parameter_block, map_intrinsics.at(indexCam).size());
        if (options.intrinsics_opt == Intrinsic_Parameter_Type::NONE)
        {
          // set the whole parameter block as constant for best performance
          problem.SetParameterBlockConstant(parameter_block);
        }
        else
        {
          const std::vector<int> vec_constant_intrinsic =
            intrinsic_it.second->subsetParameterization(options.intrinsics_opt);
          if (!vec_constant_intrinsic.empty())
          {
            ceres::SubsetParameterization *subset_parameterization =
              new ceres::SubsetParameterization(
                map_intrinsics.at(indexCam).size(), vec_constant_intrinsic);
            problem.SetParameterization(parameter_block, subset_parameterization);
          }
        }
      }
    }
    else
    {
      std::cerr << "Unsupported camera type." << std::endl;
    }
  }

  if (options.use_lines_opt){
     for (const auto& lineCorrespondence: potential_matches)
      {
      IndexT indexLine = lineCorrespondence.first;
      Eigen::Vector6d curLine3DEndpoints = all_3d_lines[indexLine];
      map_lines[indexLine] = {curLine3DEndpoints(0), curLine3DEndpoints(1), curLine3DEndpoints(2),curLine3DEndpoints(3), curLine3DEndpoints(4),curLine3DEndpoints(5)};
    
      double * parameter_block = &map_lines.at(indexLine)[0];

      problem.AddParameterBlock(parameter_block,6);

      if (options.line_opt == Line_Parameter_Type::NONE){
        problem.SetParameterBlockConstant(parameter_block);
      }
    }
  }

  // Set a LossFunction to be less penalized by false measurements
  //  - set it to nullptr if you don't want use a lossFunction.
  ceres::LossFunction * p_LossFunction =
    ceres_options_.bUse_loss_function_ ?
      new ceres::HuberLoss(Square(4.0))
      : nullptr;

  // For all visibility add reprojections errors:
  for (auto & structure_landmark_it : sfm_data.structure)
  {
    const Observations & obs = structure_landmark_it.second.obs;

    for (const auto & obs_it : obs)
    {
      // Build the residual block corresponding to the track observation:
      const View * view = sfm_data.views.at(obs_it.first).get();

      // Each Residual block takes a point and a camera as input and outputs a 2
      // dimensional residual. Internally, the cost function stores the observed
      // image location and compares the reprojection against the observation.
      ceres::CostFunction* cost_function =
        IntrinsicsToCostFunction(sfm_data.intrinsics.at(view->id_intrinsic).get(),
                                 obs_it.second.x);  

  
      if (cost_function)
      {
        if (!map_intrinsics.at(view->id_intrinsic).empty())
        {
          problem.AddResidualBlock(cost_function,
            p_LossFunction,
            &map_intrinsics.at(view->id_intrinsic)[0],
            &map_poses.at(view->id_pose)[0],
            structure_landmark_it.second.X.data());
        }
        else
        {
          problem.AddResidualBlock(cost_function,
            p_LossFunction,
            &map_poses.at(view->id_pose)[0],
            structure_landmark_it.second.X.data());
        }
      }
      else
      {
        std::cerr << "Cannot create a CostFunction for this camera model." << std::endl;
        return false;
      }
      
    }
    if (options.structure_opt == Structure_Parameter_Type::NONE)
      problem.SetParameterBlockConstant(structure_landmark_it.second.X.data());
  }

  if (options.control_point_opt.bUse_control_points)
  {
    // Use Ground Control Point:
    // - fixed 3D points with weighted observations
    for (auto & gcp_landmark_it : sfm_data.control_points)
    {
      const Observations & obs = gcp_landmark_it.second.obs;

      for (const auto & obs_it : obs)
      {
        // Build the residual block corresponding to the track observation:
        const View * view = sfm_data.views.at(obs_it.first).get();

        // Each Residual block takes a point and a camera as input and outputs a 2
        // dimensional residual. Internally, the cost function stores the observed
        // image location and compares the reprojection against the observation.
        ceres::CostFunction* cost_function =
          IntrinsicsToCostFunction(
            sfm_data.intrinsics.at(view->id_intrinsic).get(),
            obs_it.second.x,
            options.control_point_opt.weight);

        if (cost_function)
        {
          if (!map_intrinsics.at(view->id_intrinsic).empty())
          {
            problem.AddResidualBlock(cost_function,
                                     nullptr,
                                     &map_intrinsics.at(view->id_intrinsic)[0],
                                     &map_poses.at(view->id_pose)[0],
                                     gcp_landmark_it.second.X.data());
          }
          else
          {
            problem.AddResidualBlock(cost_function,
                                     nullptr,
                                     &map_poses.at(view->id_pose)[0],
                                     gcp_landmark_it.second.X.data());
          }
        }
      }
      if (obs.empty())
      {
        std::cerr
          << "Cannot use this GCP id: " << gcp_landmark_it.first
          << ". There is not linked image observation." << std::endl;
      }
      else
      {
        // Set the 3D point as FIXED (it's a valid GCP)
        problem.SetParameterBlockConstant(gcp_landmark_it.second.X.data());
      }
    }
  }

  if (options.use_lines_opt){
        //Warning !! Needs to be changed
    for (const auto& lineCorrespondence: potential_matches)
    {
      IndexT indexLine = lineCorrespondence.first;

      for (const auto& line_it: lineCorrespondence.second){

          IndexT curViewId = all_2d_lines.at(line_it).first;
          IndexT curPoseId = sfm_data.views.at(curViewId)->id_pose;
          IndexT curIntrId = sfm_data.views.at(curViewId)->id_intrinsic;

          Eigen::Matrix<double,4,1> cur_2d_line = all_2d_lines.at(line_it).second;
          double curLine[4] = {cur_2d_line(0), cur_2d_line(1), cur_2d_line(2), cur_2d_line(3)};

          ceres::CostFunction * cost_function_lines = 
          new ceres::AutoDiffCostFunction<LineReprojectionConstraintCostFunction,2,3,6,6>(
            new LineReprojectionConstraintCostFunction(curLine));

          problem.AddResidualBlock(
          cost_function_lines,
          p_LossFunction,
          &map_intrinsics.at(curIntrId)[0],
          &map_poses.at(curPoseId)[0],
          &map_lines.at(indexLine)[0]
        );
      }
    }   
  }

  // Add Pose prior constraints if any
  if (b_usable_prior)
  {
    for (const auto & view_it : sfm_data.GetViewsPriors())
    {
      const sfm::ViewPriors * prior = view_it.second.get();
      if (prior != nullptr && prior->b_use_pose_center_ && sfm_data.IsPoseAndIntrinsicDefined(prior))
      {
        // std::cout << "Will use prior constraints" << std::endl;
        // Add the cost functor (distance from Pose prior to the SfM_Data Pose center)
        ceres::CostFunction * cost_function =
          new ceres::AutoDiffCostFunction<PoseCenterConstraintCostFunction, 3, 6>(
            new PoseCenterConstraintCostFunction(prior->pose_center_, prior->center_weight_));

        problem.AddResidualBlock(
          cost_function,
          new ceres::HuberLoss(
            Square(pose_center_robust_fitting_error)),
                   &map_poses.at(prior->id_view)[0]);
      
        
      }
    }
  }
    // Configure a BA engine and run it
  //  Make Ceres automatically detect the bundle structure.
  ceres::Solver::Options ceres_config_options;
  ceres_config_options.max_num_iterations = 500;
  ceres_config_options.preconditioner_type =
    static_cast<ceres::PreconditionerType>(ceres_options_.preconditioner_type_);
  ceres_config_options.linear_solver_type =
    static_cast<ceres::LinearSolverType>(ceres_options_.linear_solver_type_);
  ceres_config_options.sparse_linear_algebra_library_type =
    static_cast<ceres::SparseLinearAlgebraLibraryType>(ceres_options_.sparse_linear_algebra_library_type_);
  ceres_config_options.minimizer_progress_to_stdout = false;
  // ceres_config_options.logging_type = ceres::SILENT;
  ceres_config_options.num_threads = ceres_options_.nb_threads_;
#if CERES_VERSION_MAJOR < 2
  ceres_config_options.num_linear_solver_threads = ceres_options_.nb_threads_;
#endif
  ceres_config_options.parameter_tolerance = ceres_options_.parameter_tolerance_;

  // Solve BA
  ceres::Solver::Summary summary;
  ceres::Solve(ceres_config_options, &problem, &summary);
  if (ceres_options_.bCeres_summary_)
    std::cout << summary.FullReport() << std::endl;

  // If no error, get back refined parameters
  if (!summary.IsSolutionUsable())
  {
    if (ceres_options_.bVerbose_)
      std::cout << "Bundle Adjustment failed." << std::endl;
    return false;
  }
  else // Solution is usable
  {
    if (ceres_options_.bVerbose_)
    {
      // Display statistics about the minimization
      std::cout << std::endl
        << "Bundle Adjustment statistics (approximated RMSE):\n"
        << " #views: " << sfm_data.views.size() << "\n"
        << " #poses: " << sfm_data.poses.size() << "\n"
        << " #intrinsics: " << sfm_data.intrinsics.size() << "\n"
        << " #tracks: " << sfm_data.structure.size() << "\n"
        << " #residuals: " << summary.num_residuals << "\n"
        << " Initial RMSE: " << std::sqrt( summary.initial_cost / summary.num_residuals) << "\n"
        << " Final RMSE: " << std::sqrt( summary.final_cost / summary.num_residuals) << "\n"
        << " Time (s): " << summary.total_time_in_seconds << "\n"
        << std::endl;
      if (options.use_motion_priors_opt)
        std::cout << "Usable motion priors: " << (int)b_usable_prior << std::endl;
    }

    // Update camera poses with refined data
    if (options.extrinsics_opt != Extrinsic_Parameter_Type::NONE)
    {
      for (auto & pose_it : sfm_data.poses)
      {
        const IndexT indexPose = pose_it.first;

        Mat3 R_refined;
        ceres::AngleAxisToRotationMatrix(&map_poses.at(indexPose)[0], R_refined.data());
        Vec3 t_refined(map_poses.at(indexPose)[3], map_poses.at(indexPose)[4], map_poses.at(indexPose)[5]);
        // Update the pose
        Pose3 & pose = pose_it.second;
        // pose = Pose3(R_refined, -R_refined.transpose() * t_refined);
         if (options.extrinsics_opt == Extrinsic_Parameter_Type::ADJUST_ROTATION)
        {
            // Update only rotation
            pose.rotation() = R_refined;
        }
        else if (options.extrinsics_opt == Extrinsic_Parameter_Type::ADJUST_TRANSLATION)
        {
            // Update only translation
            const Vec3 C_refined = -R_refined.transpose() * t_refined;
            pose.center() = C_refined;
        }
        else
        {
            // Update rotation + translation
            pose = Pose3(R_refined, -R_refined.transpose() * t_refined);
        }
      }
    }

    // Update camera intrinsics with refined data
    if (options.intrinsics_opt != Intrinsic_Parameter_Type::NONE)
    {
      for (auto & intrinsic_it : sfm_data.intrinsics)
      {
        const IndexT indexCam = intrinsic_it.first;

        const std::vector<double> & vec_params = map_intrinsics.at(indexCam);
        intrinsic_it.second->updateFromParams(vec_params);
      }
    }

    // Structure is already updated directly if needed (no data wrapping)

    if (b_usable_prior)
    {
      // set back to the original scene centroid
      openMVG::sfm::ApplySimilarity(sim_to_center.inverse(), sfm_data, true);

      //--
      // - Compute some fitting statistics
      //--

      // Collect corresponding camera centers
      std::vector<Vec3> X_SfM, X_GPS;
      for (const auto & view_it : sfm_data.GetViewsPriors())
      {
        const sfm::ViewPriors * prior = view_it.second.get();
        if (prior != nullptr && prior->b_use_pose_center_ && sfm_data.IsPoseAndIntrinsicDefined(prior))
        {
          X_SfM.push_back( sfm_data.GetPoses().at(prior->id_pose).center() );
          X_GPS.push_back( prior->pose_center_ );
        }
      }
      // Compute the registration fitting error (once BA with Prior have been used):
      if (X_GPS.size() > 3)
      {
        // Compute the median residual error
        Vec residual = (Eigen::Map<Mat3X>(X_SfM[0].data(), 3, X_SfM.size()) - Eigen::Map<Mat3X>(X_GPS[0].data(), 3, X_GPS.size())).colwise().norm();
        std::cout
          << "Pose prior statistics (user units):\n"
          << " - Starting median fitting error: " << pose_center_robust_fitting_error << "\n"
          << " - Final fitting error:";
        minMaxMeanMedian<Vec::Scalar>(residual.data(), residual.data() + residual.size());
      }
    }
    return true;
  }
}

} // namespace sfm
} // namespace openMVG
