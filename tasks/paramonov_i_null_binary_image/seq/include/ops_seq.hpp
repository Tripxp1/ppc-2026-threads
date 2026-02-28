#pragma once

#include <cstddef>
#include <vector>

#include "paramonov_i_null_binary_image_seq/common/include/common.hpp"
#include "task/include/task.hpp"

namespace paramonov_i_null_binary_image_seq {

class ParamonovINullBinaryImageSeq : public BaseTask {
public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit ParamonovINullBinaryImageSeq(const InType &in);

private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void BinarizeImage();
  void ExtractComponents();
  static std::vector<Point> ComputeConvexHull(const std::vector<Point> &points);
  static size_t GetIndex(int x, int y, int width);
  void ExploreComponent(int start_col, int start_row, int width, int height,
                        std::vector<bool> &visited,
                        std::vector<Point> &component);

  BinaryImage work_image_;
};

} // namespace paramonov_i_null_binary_image_seq