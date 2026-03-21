#include "paramonov_v_bin_img_conv_hul_omp/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <stack>
#include <utility>
#include <vector>

#include "paramonov_v_bin_img_conv_hul_omp/common/include/common.hpp"

namespace paramonov_v_bin_img_conv_hul_omp {

namespace {
constexpr std::array<std::pair<int, int>, 4> kNeighbors = {{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};

bool ComparePoints(const PixelPoint &a, const PixelPoint &b) {
  if (a.row != b.row) {
    return a.row < b.row;
  }
  return a.col < b.col;
}

}  // namespace

ConvexHullOMP::ConvexHullOMP(const InputType &input) {
  SetTypeOfTask(StaticTaskType());
  GetInput() = input;
}

bool ConvexHullOMP::ValidationImpl() {
  const auto &img = GetInput();
  if (img.rows <= 0 || img.cols <= 0) {
    return false;
  }

  const size_t expected_size = static_cast<size_t>(img.rows) * static_cast<size_t>(img.cols);
  return img.pixels.size() == expected_size;
}

bool ConvexHullOMP::PreProcessingImpl() {
  working_image_ = GetInput();
  BinarizeImage();
  GetOutput().clear();
  return true;
}

bool ConvexHullOMP::RunImpl() {
  ExtractConnectedComponents();
  return true;
}

bool ConvexHullOMP::PostProcessingImpl() {
  return true;
}

void ConvexHullOMP::BinarizeImage(uint8_t threshold) {
  const size_t pixel_count = working_image_.pixels.size();

#pragma omp parallel for
  for (size_t i = 0; i < pixel_count; ++i) {
    working_image_.pixels[i] = working_image_.pixels[i] > threshold ? uint8_t{255} : uint8_t{0};
  }
}

void ConvexHullOMP::FloodFill(int start_row, int start_col, std::vector<bool> &visited,
                              std::vector<PixelPoint> &component) const {
  std::stack<PixelPoint> pixel_stack;
  pixel_stack.emplace(start_row, start_col);

  const int rows = working_image_.rows;
  const int cols = working_image_.cols;

  visited[PixelIndex(start_row, start_col, cols)] = true;

  while (!pixel_stack.empty()) {
    PixelPoint current = pixel_stack.top();
    pixel_stack.pop();

    component.push_back(current);

    for (const auto &[dr, dc] : kNeighbors) {
      int next_row = current.row + dr;
      int next_col = current.col + dc;

      if (next_row >= 0 && next_row < rows && next_col >= 0 && next_col < cols) {
        size_t idx = PixelIndex(next_row, next_col, cols);
        if (!visited[idx] && working_image_.pixels[idx] == 255) {
          visited[idx] = true;
          pixel_stack.emplace(next_row, next_col);
        }
      }
    }
  }
}

void ConvexHullOMP::ExtractConnectedComponents() {
  const int rows = working_image_.rows;
  const int cols = working_image_.cols;
  const size_t total_pixels = static_cast<size_t>(rows) * static_cast<size_t>(cols);

  std::vector<bool> visited(total_pixels, false);

  // Вектор для хранения оболочек каждого потока
  // thread_hulls[thread_id] - это вектор оболочек для потока thread_id
  std::vector<std::vector<std::vector<PixelPoint>>> thread_hulls;

#pragma omp parallel
  {
#pragma omp single
    {
      thread_hulls.resize(omp_get_num_threads());
    }

    int thread_id = omp_get_thread_num();
    auto &local_hulls = thread_hulls[thread_id];  // local_hulls - это vector<vector<PixelPoint>>

#pragma omp for schedule(dynamic, 64)
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        size_t idx = PixelIndex(row, col, cols);

        bool need_process = false;
#pragma omp critical
        {
          if (working_image_.pixels[idx] == 255 && !visited[idx]) {
            visited[idx] = true;
            need_process = true;
          }
        }

        if (need_process) {
          std::vector<PixelPoint> component;
          FloodFill(row, col, visited, component);

          if (!component.empty()) {
            std::vector<PixelPoint> hull = ComputeConvexHull(component);
            local_hulls.push_back(std::move(hull));  // добавляем оболочку в вектор оболочек потока
          }
        }
      }
    }
  }

  // Собираем результаты из всех потоков
  auto &output_hulls = GetOutput();
  for (auto &thread_hull : thread_hulls) {
    output_hulls.insert(output_hulls.end(), std::make_move_iterator(thread_hull.begin()),
                        std::make_move_iterator(thread_hull.end()));
  }
}

int64_t ConvexHullOMP::Orientation(const PixelPoint &p, const PixelPoint &q, const PixelPoint &r) {
  return (static_cast<int64_t>(q.col - p.col) * (r.row - p.row)) -
         (static_cast<int64_t>(q.row - p.row) * (r.col - p.col));
}

std::vector<PixelPoint> ConvexHullOMP::ComputeConvexHull(const std::vector<PixelPoint> &points) {
  if (points.size() <= 2) {
    return points;
  }

  auto lowest_point = *std::ranges::min_element(points, ComparePoints);

  std::vector<PixelPoint> sorted_points;
  std::ranges::copy_if(points, std::back_inserter(sorted_points), [&lowest_point](const PixelPoint &p) {
    return (p.row != lowest_point.row) || (p.col != lowest_point.col);
  });

  std::ranges::sort(sorted_points, [&lowest_point](const PixelPoint &a, const PixelPoint &b) {
    int64_t orient = Orientation(lowest_point, a, b);
    if (orient == 0) {
      int64_t dist_a = ((a.row - lowest_point.row) * (a.row - lowest_point.row)) +
                       ((a.col - lowest_point.col) * (a.col - lowest_point.col));
      int64_t dist_b = ((b.row - lowest_point.row) * (b.row - lowest_point.row)) +
                       ((b.col - lowest_point.col) * (b.col - lowest_point.col));
      return dist_a < dist_b;
    }
    return orient > 0;
  });

  std::vector<PixelPoint> hull;
  hull.push_back(lowest_point);

  for (const auto &p : sorted_points) {
    while (hull.size() >= 2) {
      const auto &a = hull[hull.size() - 2];
      const auto &b = hull.back();

      if (Orientation(a, b, p) <= 0) {
        hull.pop_back();
      } else {
        break;
      }
    }
    hull.push_back(p);
  }

  return hull;
}

}  // namespace paramonov_v_bin_img_conv_hul_omp
