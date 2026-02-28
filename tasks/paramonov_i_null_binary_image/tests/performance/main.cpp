#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "paramonov_i_null_binary_image/common/include/common.hpp"
#include "paramonov_i_null_binary_image/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace paramonov_i_null_binary_image {

class ParamonovINullBinaryImagePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr size_t kImageSize = 512;

 protected:
  void SetUp() override {
    test_image_.width = static_cast<int>(kImageSize);
    test_image_.height = static_cast<int>(kImageSize);
    test_image_.pixels.assign(kImageSize * kImageSize, 0);

    // Generate random foreground pixels
    for (size_t i = 0; i < kImageSize; ++i) {
      size_t idx1 = (i * kImageSize) + ((i * 13) % kImageSize);
      size_t idx2 = (i * kImageSize) + ((i * 29 + 7) % kImageSize);
      test_image_.pixels[idx1] = 255;
      test_image_.pixels[idx2] = 255;
    }
  }

  bool CheckTestOutputData(OutType &out) override {
    return !out.convex_hulls.empty();
  }

  InType GetTestInputData() override {
    return test_image_;
  }

 private:
  InType test_image_;
};

TEST_P(ParamonovINullBinaryImagePerfTests, RunPerf) {
  ExecuteTest(GetParam());
}

namespace {

const auto kPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ParamonovINullBinaryImageSeq>(PPC_SETTINGS_paramonov_i_null_binary_image);

const auto kValues = ppc::util::TupleToGTestValues(kPerfTasks);

INSTANTIATE_TEST_SUITE_P(PerformanceTests, ParamonovINullBinaryImagePerfTests, kValues,
                         ParamonovINullBinaryImagePerfTests::CustomPerfTestName);

}  // namespace

}  // namespace paramonov_i_null_binary_image
