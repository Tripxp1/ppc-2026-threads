#include "paramonov_i_null_binary_image_seq/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <ranges>
#include <utility>
#include <vector>

#include "paramonov_i_null_binary_image_seq/common/include/common.hpp"

namespace paramonov_i_null_binary_image_seq {

namespace {

constexpr uint8_t kBinaryThreshold = 128;
constexpr std::array<std::pair<int, int>, 4> kNeighborOffsets = {
    std::make_pair(1, 0), std::make_pair(-1, 0), std::make_pair(0, 1),
    std::make_pair(0, -1)};

int64_t ComputeCrossProduct(const Point &a, const Point &b, const Point &c) {
  int64_t abx = static_cast<int64_t>(b.x) - static_cast<int64_t>(a.x);
  int64_t aby = static_cast<int64_t>(b.y) - static_cast<int64_t>(a.y);
  int64_t bcx = static_cast<int64_t>(c.x) - static_cast<int64_t>(b.x);
  int64_t bcy = static_cast<int64_t>(c.y) - static_cast<int64_t>(b.y);
  return (abx * bcy) - (aby * bcx);
}

bool IsForegroundPixel(uint8_t pixel) { return pixel > kBinaryThreshold; }

bool IsValidCoordinate(int x, int y, int width, int height) {
  return x >= 0 && x < width && y >= 0 && y < height;
}

} // namespace

ParamonovINullBinaryImageSeq::ParamonovINullBinaryImageSeq(const InType &in)
    : work_image_(in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool ParamonovINullBinaryImageSeq::ValidationImpl() {
  const auto &in = GetInput();
  bool valid_dimensions = in.width > 0 && in.height > 0;
  bool valid_size = in.pixels.size() == static_cast<size_t>(in.width) *
                                            static_cast<size_t>(in.height);
  return valid_dimensions && valid_size;
}

bool ParamonovINullBinaryImageSeq::PreProcessingImpl() {
  work_image_ = GetInput();
  BinarizeImage();
  return true;
}

bool ParamonovINullBinaryImageSeq::RunImpl() {
  ExtractComponents();

  work_image_.convex_hulls.clear();
  work_image_.convex_hulls.reserve(work_image_.components.size());

  for (const auto &component : work_image_.components) {
    if (component.empty()) {
      continue;
    }
    if (component.size() <= 2) {
      work_image_.convex_hulls.push_back(component);
    } else {
      work_image_.convex_hulls.push_back(ComputeConvexHull(component));
    }
  }

  GetOutput() = work_image_;
  return true;
}

bool ParamonovINullBinaryImageSeq::PostProcessingImpl() { return true; }

size_t ParamonovINullBinaryImageSeq::GetIndex(int x, int y, int width) {
  return (static_cast<size_t>(y) * static_cast<size_t>(width)) +
         static_cast<size_t>(x);
}

void ParamonovINullBinaryImageSeq::BinarizeImage() {
  for (auto &pixel : work_image_.pixels) {
    pixel = IsForegroundPixel(pixel) ? static_cast<uint8_t>(255)
                                     : static_cast<uint8_t>(0);
  }
}

void ParamonovINullBinaryImageSeq::ExploreComponent(
    int start_col, int start_row, int width, int height,
    std::vector<bool> &visited, std::vector<Point> &component) {
  std::queue<Point> queue;
  queue.emplace(start_col, start_row);
  visited[GetIndex(start_col, start_row, width)] = true;

  while (!queue.empty()) {
    Point current = queue.front();
    queue.pop();
    component.push_back(current);

    for (auto [dx, dy] : kNeighborOffsets) {
      int next_x = current.x + dx;
      int next_y = current.y + dy;

      if (!IsValidCoordinate(next_x, next_y, width, height)) {
        continue;
      }

      size_t next_idx = GetIndex(next_x, next_y, width);
      if (visited[next_idx] || work_image_.pixels[next_idx] == 0) {
        continue;
      }

      visited[next_idx] = true;
      queue.emplace(next_x, next_y);
    }
  }
}

void ParamonovINullBinaryImageSeq::ExtractComponents() {
  int width = work_image_.width;
  int height = work_image_.height;

  std::vector<bool> visited(
      static_cast<size_t>(width) * static_cast<size_t>(height), false);
  work_image_.components.clear();

  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      size_t idx = GetIndex(col, row, width);
      if (work_image_.pixels[idx] == 0 || visited[idx]) {
        continue;
      }

      std::vector<Point> component;
      ExploreComponent(col, row, width, height, visited, component);

      if (!component.empty()) {
        work_image_.components.push_back(std::move(component));
      }
    }
  }
}

std::vector<Point> ParamonovINullBinaryImageSeq::ComputeConvexHull(
    const std::vector<Point> &points) {
  if (points.size() <= 2) {
    return points;
  }

  std::vector<Point> sorted_points = points;
  std::ranges::sort(sorted_points, [](const Point &a, const Point &b) {
    return (a.x != b.x) ? (a.x < b.x) : (a.y < b.y);
  });

  auto [first_duplicate, last_duplicate] = std::ranges::unique(sorted_points);
  sorted_points.erase(first_duplicate, last_duplicate);

  if (sorted_points.size() <= 2) {
    return sorted_points;
  }

  std::vector<Point> lower_hull;
  std::vector<Point> upper_hull;
  lower_hull.reserve(sorted_points.size());
  upper_hull.reserve(sorted_points.size());

  // Build lower part of hull
  for (const auto &point : sorted_points) {
    while (lower_hull.size() >= 2 &&
           ComputeCrossProduct(lower_hull[lower_hull.size() - 2],
                               lower_hull.back(), point) <= 0) {
      lower_hull.pop_back();
    }
    lower_hull.push_back(point);
  }

  // Build upper part of hull
  for (const auto &point : std::ranges::reverse_view(sorted_points)) {
    while (upper_hull.size() >= 2 &&
           ComputeCrossProduct(upper_hull[upper_hull.size() - 2],
                               upper_hull.back(), point) <= 0) {
      upper_hull.pop_back();
    }
    upper_hull.push_back(point);
  }

  // Combine both parts (remove duplicate endpoints)
  lower_hull.pop_back();
  upper_hull.pop_back();
  lower_hull.insert(lower_hull.end(), upper_hull.begin(), upper_hull.end());

  return lower_hull;
}

} // namespace paramonov_i_null_binary_image_seq