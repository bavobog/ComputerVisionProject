#include "MapManager.h" // or "MapManager.hpp"
#include <cmath>

MapManager::MapManager(int max_scans) : max_scans_(max_scans) {}

void MapManager::addScan(const std::vector<Point3D>& scan_in_world) {
    scan_buffer_.push_back(scan_in_world);

    if (static_cast<int>(scan_buffer_.size()) > max_scans_) {
        scan_buffer_.pop_front();
    }

    flat_map_points_.clear();
    size_t total_size = 0;
    for (const auto& scan : scan_buffer_) {
        total_size += scan.size();
    }
    flat_map_points_.reserve(total_size);

    for (const auto& scan : scan_buffer_) {
        flat_map_points_.insert(flat_map_points_.end(), scan.begin(), scan.end());
    }

    rebuildKDTree();
}

void MapManager::rebuildKDTree() {
    if (flat_map_points_.empty()) return;

    adaptor_ = std::make_unique<MapPointCloudAdaptor>(MapPointCloudAdaptor{flat_map_points_});
    kd_tree_ = std::make_unique<MapKDTree>(
        3, *adaptor_, nanoflann::KDTreeSingleIndexAdaptorParams(10)
    );
    kd_tree_->buildIndex();
}

bool MapManager::hasCloseNeighbor(const Eigen::Vector3d& query_pt, double r_max) const {
    if (!kd_tree_ || flat_map_points_.empty()) return true;

    size_t out_idx;
    double out_dist_sq;
    nanoflann::KNNResultSet<double, size_t> resultSet(1);
    resultSet.init(&out_idx, &out_dist_sq);

    kd_tree_->findNeighbors(resultSet, query_pt.data());

    return (std::sqrt(out_dist_sq) <= r_max);
}

void MapManager::queryRadius(const Eigen::Vector3d& query_pt, double r,
                              std::vector<Point3D>& out_neighbors) const
{
    out_neighbors.clear();
    if (!kd_tree_ || flat_map_points_.empty()) return;

    std::vector<nanoflann::ResultItem<size_t, double>> indices_dists;
    nanoflann::SearchParameters params;
    params.sorted = false;

    const double radius_sq = r * r;
    kd_tree_->radiusSearch(query_pt.data(), radius_sq, indices_dists, params);

    out_neighbors.reserve(indices_dists.size());
    for (const auto& item : indices_dists) {
        out_neighbors.push_back(flat_map_points_[item.first]);
    }
}