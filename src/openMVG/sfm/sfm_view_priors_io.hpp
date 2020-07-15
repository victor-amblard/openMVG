// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2016 Pierre Moulon.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef OPENMVG_SFM_SFM_VIEW_PRIORS_IO_HPP
#define OPENMVG_SFM_SFM_VIEW_PRIORS_IO_HPP

#include "openMVG/sfm/sfm_view_priors.hpp"

#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>

template <class Archive>
void openMVG::sfm::ViewPriors::save( Archive & ar ) const
{
  View::save(ar);

  // Pose center prior
  if (b_use_pose_center_)
  {
    ar( cereal::make_nvp( "use_pose_center_prior", b_use_pose_center_ ) );
    const std::vector<double> vec_weights { center_weight_( 0 ), center_weight_( 1 ), center_weight_( 2 ) };
    ar( cereal::make_nvp( "center_weight", vec_weights ) );
    const std::vector<double> vec { pose_center_( 0 ), pose_center_( 1 ), pose_center_( 2 ) };
    ar( cereal::make_nvp( "center", vec ) );
  }

  // Pose rotation prior
  /*
  if (b_use_pose_rotation_)
  {
    ar( cereal::make_nvp( "use_pose_rotation_prior", b_use_pose_rotation_ ) );
    ar( cereal::make_nvp( "rotation_weight", rotation_weight_ ) );
    const std::vector<std::vector<double>> mat =
    {
      { pose_rotation_( 0, 0 ), pose_rotation_( 0, 1 ), pose_rotation_( 0, 2 ) },
      { pose_rotation_( 1, 0 ), pose_rotation_( 1, 1 ), pose_rotation_( 1, 2 ) },
      { pose_rotation_( 2, 0 ), pose_rotation_( 2, 1 ), pose_rotation_( 2, 2 ) }
    };
    ar( cereal::make_nvp( "rotation", mat ) );
  }
  */
}

template <class Archive>
void openMVG::sfm::ViewPriors::load( Archive & ar )
{

  // View::load(ar);
  // std::cout << "Loading view prior!" << std::endl;
  // ar(cereal::base_class<View>(this));
    // ar( cereal::make_nvp( "use_pose_center_prior", data ) );

  // Pose center prior
  // std::shared_ptr<View> v = std::make_shared<View>();
  // ar(cereal::base_class<View>(this), this);

  try
  {
      std::string local_path = s_Img_path;
  std::string filename = s_Img_path;
  bool usePosePrior;
     ar(cereal::make_nvp("local_path", local_path),
     cereal::make_nvp("filename", filename),
     cereal::make_nvp("width", ui_width),
     cereal::make_nvp("height", ui_height),
     cereal::make_nvp("id_view", id_view),
     cereal::make_nvp("id_intrinsic", id_intrinsic),
     cereal::make_nvp("id_pose", id_pose),
     cereal::make_nvp("lidar_filename", s_Lidar_path));
     ar(cereal::make_nvp("use_pose_center_prior", usePosePrior));
    std::vector<double> vec( 3 );
    ar( cereal::make_nvp( "center_weight", vec ) );
    center_weight_ = Eigen::Map<const Vec3>( &vec[0] );
    ar( cereal::make_nvp( "center", vec ) );
    pose_center_ = Eigen::Map<const Vec3>( &vec[0] );
    if (usePosePrior)
      SetPoseCenterPrior(pose_center_, center_weight_);

  }
  catch ( cereal::Exception & e )
  {
    std::cout << "failed to load prior" << std::endl;
    // if it fails just use a default settings
    b_use_pose_center_ = false;
  }

  // Pose rotation prior
  /*
  try
  {
    ar( cereal::make_nvp( "use_pose_rotation_prior", b_use_pose_rotation_ ) );
    ar( cereal::make_nvp( "rotation_weight", rotation_weight_ ) );
    std::vector<std::vector<double>> mat( 3, std::vector<double>( 3 ) );
    ar( cereal::make_nvp( "rotation", mat ) );
    // copy back to the rotation
    pose_rotation_.row( 0 ) = Eigen::Map<const Vec3>( &( mat[0][0] ) );
    pose_rotation_.row( 1 ) = Eigen::Map<const Vec3>( &( mat[1][0] ) );
    pose_rotation_.row( 2 ) = Eigen::Map<const Vec3>( &( mat[2][0] ) );
  }
  catch ( const cereal::Exception & e )
  {
    // if it fails just use a default settings
    b_use_pose_rotation_ = false;
  }
  */
}

CEREAL_REGISTER_TYPE_WITH_NAME( openMVG::sfm::ViewPriors, "view_priors" );
CEREAL_REGISTER_POLYMORPHIC_RELATION(openMVG::sfm::View, openMVG::sfm::ViewPriors);

#endif // OPENMVG_SFM_SFM_VIEW_PRIORS_IO_HPP
