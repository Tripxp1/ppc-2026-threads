#pragma once

#include "paramonov_i_null_binary_image/common/include/common.hpp"
#include "task/include/task.hpp"

namespace paramonov_i_null_binary_image {

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

}  // namespace paramonov_i_null_binary_image
