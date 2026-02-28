#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "paramonov_i_constr_hull_for_binary_image_seq/common/include/common.hpp"
#include "paramonov_i_constr_hull_for_binary_image_seq/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace paramonov_i_constr_hull_for_binary_image_seq {

class ParamonovIConstrHullPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr size_t kSize = 512;

 protected:
  void SetUp() override {
    input_.width = static_cast<int>(kSize);
    input_.height = static_cast<int>(kSize);
    input_.pixels.assign(kSize * kSize, 0);

    for (size_t i = 0; i < kSize; ++i) {
      size_t idx1 = (i * kSize) + ((i * 13) % kSize);
      size_t idx2 = (i * kSize) + ((i * 29 + 7) % kSize);
      input_.pixels[idx1] = 255;
      input_.pixels[idx2] = 255;
    }
  }

  bool CheckTestOutputData(OutType &out) override {
    return !out.convex_hulls.empty();
  }

  InType GetTestInputData() override {
    return input_;
  }

 private:
  InType input_;
};

TEST_P(ParamonovIConstrHullPerfTests, RunPerf) {
  ExecuteTest(GetParam());
}

namespace {

const auto kPerfTasks = ppc::util::MakeAllPerfTasks<InType, ParamonovIConstrHullSeq>(
    PPC_SETTINGS_paramonov_i_constr_hull_for_binary_image_seq);

const auto kValues = ppc::util::TupleToGTestValues(kPerfTasks);

INSTANTIATE_TEST_SUITE_P(Perf, ParamonovIConstrHullPerfTests, kValues,
                         ParamonovIConstrHullPerfTests::CustomPerfTestName);

}  // namespace

}  // namespace paramonov_i_constr_hull_for_binary_image_seq
