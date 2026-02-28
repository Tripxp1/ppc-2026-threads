#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "paramonov_i_null_binary_image_seq/common/include/common.hpp"
#include "paramonov_i_null_binary_image_seq/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace paramonov_i_null_binary_image_seq {

namespace {

struct TestCase {
  BinaryImage image;
  std::vector<std::vector<Point>> expected_hulls;
};

BinaryImage CreateImage(int width, int height, uint8_t value = 0) {
  BinaryImage img;
  img.width = width;
  img.height = height;
  img.pixels.assign(static_cast<size_t>(width) * static_cast<size_t>(height),
                    value);
  return img;
}

void SetPixelValue(BinaryImage &img, int col, int row, uint8_t value) {
  size_t idx = (static_cast<size_t>(row) * static_cast<size_t>(img.width)) +
               static_cast<size_t>(col);
  img.pixels[idx] = value;
}

TestCase CreateTestCase1() {
  TestCase tc;
  tc.image = CreateImage(5, 5);
  SetPixelValue(tc.image, 2, 2, 200);
  tc.expected_hulls = {{{2, 2}}};
  return tc;
}

TestCase CreateTestCase2() {
  TestCase tc;
  tc.image = CreateImage(6, 6);
  SetPixelValue(tc.image, 1, 1, 255);
  SetPixelValue(tc.image, 4, 4, 255);
  tc.expected_hulls = {{{1, 1}}, {{4, 4}}};
  return tc;
}

TestCase CreateTestCase3() {
  TestCase tc;
  tc.image = CreateImage(7, 3);
  SetPixelValue(tc.image, 2, 1, 255);
  SetPixelValue(tc.image, 3, 1, 255);
  SetPixelValue(tc.image, 4, 1, 255);
  tc.expected_hulls = {{{2, 1}, {4, 1}}};
  return tc;
}

TestCase CreateTestCase4() {
  TestCase tc;
  tc.image = CreateImage(8, 8);
  for (int row = 2; row <= 5; ++row) {
    for (int col = 3; col <= 6; ++col) {
      SetPixelValue(tc.image, col, row, 255);
    }
  }
  tc.expected_hulls = {{{3, 2}, {6, 2}, {6, 5}, {3, 5}}};
  return tc;
}

TestCase CreateTestCase5() {
  TestCase tc;
  tc.image = CreateImage(9, 9);
  for (int row = 0; row < 9; ++row) {
    for (int col = 0; col < 9; ++col) {
      if (std::abs(col - 4) + std::abs(row - 4) <= 4) {
        SetPixelValue(tc.image, col, row, 255);
      }
    }
  }
  tc.expected_hulls = {{{0, 4}, {4, 0}, {8, 4}, {4, 8}}};
  return tc;
}

const std::vector<TestCase> &GetAllTestCases() {
  static std::vector<TestCase> cases = {CreateTestCase1(), CreateTestCase2(),
                                        CreateTestCase3(), CreateTestCase4(),
                                        CreateTestCase5()};
  return cases;
}

const TestCase &GetTestCase(int id) {
  return GetAllTestCases()[static_cast<size_t>(id)];
}

std::vector<Point> NormalizeHull(const std::vector<Point> &hull) {
  std::vector<Point> result = hull;
  std::ranges::sort(result, [](const Point &a, const Point &b) {
    return (a.y != b.y) ? (a.y < b.y) : (a.x < b.x);
  });
  auto [first, last] = std::ranges::unique(result);
  result.erase(first, last);
  return result;
}

std::vector<std::vector<Point>>
NormalizeAllHulls(const std::vector<std::vector<Point>> &hulls) {
  std::vector<std::vector<Point>> result;
  result.reserve(hulls.size());
  for (const auto &h : hulls) {
    result.push_back(NormalizeHull(h));
  }
  std::ranges::sort(result);
  return result;
}

bool CompareHullSets(const std::vector<std::vector<Point>> &a,
                     const std::vector<std::vector<Point>> &b) {
  return NormalizeAllHulls(a) == NormalizeAllHulls(b);
}

} // namespace

class ParamonovINullBinaryImageFuncTests
    : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
public:
  static std::string FormatTestParam(const TestType &p) {
    return std::to_string(std::get<0>(p)) + "_" + std::get<1>(p);
  }

protected:
  bool CheckTestOutputData(OutType &out) override {
    auto param = std::get<2>(GetParam());
    int id = std::get<0>(param);
    const auto &tc = GetTestCase(id);
    return CompareHullSets(tc.expected_hulls, out.convex_hulls);
  }

  InType GetTestInputData() override {
    auto param = std::get<2>(GetParam());
    int id = std::get<0>(param);
    return GetTestCase(id).image;
  }
};

namespace {

TEST_P(ParamonovINullBinaryImageFuncTests, Test) { ExecuteTest(GetParam()); }

const std::array<TestType, 5> kTestParams = {
    std::make_tuple(0, "case1_single_pixel"),
    std::make_tuple(1, "case2_two_pixels"),
    std::make_tuple(2, "case3_horizontal_line"),
    std::make_tuple(3, "case4_rectangle"), std::make_tuple(4, "case5_diamond")};

const auto kTasks =
    ppc::util::AddFuncTask<ParamonovINullBinaryImageSeq, InType>(
        kTestParams, PPC_SETTINGS_paramonov_i_null_binary_image_seq);

const auto kValues = ppc::util::ExpandToValues(kTasks);
const auto kName = ParamonovINullBinaryImageFuncTests::PrintFuncTestName<
    ParamonovINullBinaryImageFuncTests>;

INSTANTIATE_TEST_SUITE_P(ParamonovTests, ParamonovINullBinaryImageFuncTests,
                         kValues, kName);

} // namespace

} // namespace paramonov_i_null_binary_image_seq