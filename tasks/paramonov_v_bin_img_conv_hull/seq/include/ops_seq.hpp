#pragma once

#include "paramonov_v_bin_img_conv_hull/common/include/common.hpp"
#include "task/include/task.hpp"

namespace paramonov_v_bin_img_conv_hull {

class BinaryConvexHullSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit BinaryConvexHullSEQ(const InType &in);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace paramonov_v_bin_img_conv_hull
