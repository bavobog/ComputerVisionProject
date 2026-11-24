#include "code/FeatureExtractor.h"
#include "MapManager.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>
#include <code/nanoflann.hpp>

// Adapter for single-scan KD-Tree used during normal estimation
struct ScanPointCloudAdaptor {
    const std::vector<Point3D>& pts;

    [[nodiscard]] inline size_t kdtree_get_point_count() const { return pts.size(); }

    [[nodiscard]] inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
        return pts[idx].position[dim];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

using ScanKDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, ScanPointCloudAdaptor>,
    ScanPointCloudAdaptor, 3, uint32_t>;

FeatureExtractor::FeatureExtractor(int s_samples_per_list, double r_outlier_max)
    : s_samples_(s_samples_per_list), r_max_(r_outlier_max) {}

void FeatureExtractor::computeNormalsAndPlanarity(std::vector<Point3D>& scan_points, int k_neighbors) {
    const size_t num_points = scan_points.size();
    if (num_points < static_cast<size_t>(k_neighbors)) return;

    ScanPointCloudAdaptor adaptor{scan_points};
    ScanKDTree local_tree(3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    local_tree.buildIndex();

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(num_points); ++i) {
        std::vector<uint32_t> neighbor_indices(k_neighbors);
        std::vector<double> out_dists_sq(k_neighbors);

        local_tree.knnSearch(scan_points[i].position.data(), k_neighbors,
                             neighbor_indices.data(), out_dists_sq.data());

        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        for (int k = 0; k < k_neighbors; ++k) {
            centroid += scan_points[neighbor_indices[k]].position;
        }
        centroid /= static_cast<double>(k_neighbors);

        Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
        for (int k = 0; k < k_neighbors; ++k) {
            Eigen::Vector3d dev = scan_points[neighbor_indices[k]].position - centroid;
            covariance += dev * dev.transpose();
        }
        covariance /= static_cast<double>(k_neighbors);

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
        if (solver.info() != Eigen::Success) continue;

        Eigen::Vector3d eigenvalues = solver.eigenvalues();
        Eigen::Matrix3d eigenvectors = solver.eigenvectors();

        double l1 = std::max(0.0, eigenvalues[2]);
        double l2 = std::max(0.0, eigenvalues[1]);
        double l3 = std::max(0.0, eigenvalues[0]);

        scan_points[i].normal = eigenvectors.col(0).normalized();

        double s1 = std::sqrt(l1);
        double s2 = std::sqrt(l2);
        double s3 = std::sqrt(l3);

        scan_points[i].a2d = (s1 > 1e-6) ? ((s2 - s3) / s1) : 0.0;
    }
}

std::array<double, 9> FeatureExtractor::computeObservabilityScores(const Point3D& pt) {
    const Eigen::Vector3d& x = pt.position;
    const Eigen::Vector3d& n = pt.normal;
    double a2d_sq = pt.a2d * pt.a2d;

    Eigen::Vector3d cross = x.cross(n);

    Eigen::Vector3d X_v(1.0, 0.0, 0.0);
    Eigen::Vector3d Y_v(0.0, 1.0, 0.0);
    Eigen::Vector3d Z_v(0.0, 0.0, 1.0);

    return {
         a2d_sq * (cross.dot(X_v)),
        -a2d_sq * (cross.dot(X_v)),
         a2d_sq * (cross.dot(Y_v)),
        -a2d_sq * (cross.dot(Y_v)),
         a2d_sq * (cross.dot(Z_v)),
        -a2d_sq * (cross.dot(Z_v)),
         a2d_sq * std::abs(n.dot(X_v)),
         a2d_sq * std::abs(n.dot(Y_v)),
         a2d_sq * std::abs(n.dot(Z_v))
    };
}

std::vector<Point3D> FeatureExtractor::selectObservabilitySamples(
    const std::vector<Point3D>& scan_points,
    const MapManager& map_manager) const
{
    const size_t num_points = scan_points.size();
    if (num_points == 0) return {};

    std::vector<std::array<double, 9>> all_scores(num_points);
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(num_points); ++i) {
        all_scores[i] = computeObservabilityScores(scan_points[i]);
    }

    std::vector<Point3D> selected_samples;
    selected_samples.reserve(9 * s_samples_);

    for (int k = 0; k < 9; ++k) {
        std::vector<size_t> indices(num_points);
        std::iota(indices.begin(), indices.end(), 0);

        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            return all_scores[a][k] > all_scores[b][k];
        });

        int collected = 0;
        for (size_t idx : indices) {
            if (collected >= s_samples_) break;

            const auto& candidate = scan_points[idx];

            if (map_manager.isEmpty() || map_manager.hasCloseNeighbor(candidate.position, r_max_)) {
                selected_samples.push_back(candidate);
                collected++;
            }
        }
    }

    return selected_samples;
}