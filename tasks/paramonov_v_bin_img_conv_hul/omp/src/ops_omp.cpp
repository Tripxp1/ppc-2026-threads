#include "paramonov_v_bin_img_conv_hul/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <mutex>
#include <stack>
#include <utility>
#include <vector>

#include "paramonov_v_bin_img_conv_hul/common/include/common.hpp"

namespace paramonov_v_bin_img_conv_hul {

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
  const size_t total_pixels = working_image_.pixels.size();

#pragma omp parallel for
  for (size_t i = 0; i < total_pixels; ++i) {
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
  std::vector<std::vector<PixelPoint>> all_components;

#pragma omp parallel
  {
    std::vector<std::vector<PixelPoint>> local_components;

#pragma omp for schedule(dynamic)
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        size_t idx = PixelIndex(row, col, cols);

        if (working_image_.pixels[idx] == 255) {
          bool need_process = false;
#pragma omp critical
          {
            if (!visited[idx]) {
              visited[idx] = true;
              need_process = true;
            }
          }

          if (need_process) {
            std::vector<PixelPoint> component;
            std::stack<PixelPoint> pixel_stack;
            pixel_stack.emplace(row, col);

            const int local_rows = rows;
            const int local_cols = cols;

            while (!pixel_stack.empty()) {
              PixelPoint current = pixel_stack.top();
              pixel_stack.pop();
              component.push_back(current);

              for (const auto &[dr, dc] : kNeighbors) {
                int next_row = current.row + dr;
                int next_col = current.col + dc;

                if (next_row >= 0 && next_row < local_rows && next_col >= 0 && next_col < local_cols) {
                  size_t next_idx = PixelIndex(next_row, next_col, local_cols);
                  bool should_add = false;

#pragma omp critical
                  {
                    if (!visited[next_idx] && working_image_.pixels[next_idx] == 255) {
                      visited[next_idx] = true;
                      should_add = true;
                    }
                  }

                  if (should_add) {
                    pixel_stack.emplace(next_row, next_col);
                  }
                }
              }
            }

            if (!component.empty()) {
              local_components.push_back(std::move(component));
            }
          }
        }
      }
    }

#pragma omp critical
    {
      all_components.insert(all_components.end(), std::make_move_iterator(local_components.begin()),
                            std::make_move_iterator(local_components.end()));
    }
  }

  auto &output_hulls = GetOutput();
  output_hulls.resize(all_components.size());

#pragma omp parallel for
  for (size_t i = 0; i < all_components.size(); ++i) {
    output_hulls[i] = ComputeConvexHull(all_components[i]);
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

}  // namespace paramonov_v_bin_img_conv_hul
