#ifndef MAP_MANAGER_HPP
#define MAP_MANAGER_HPP

#include "FeatureExtractor.h"
#include <vector>
#include <deque>
#include <memory>
#include <Eigen/Dense>
#include <code/nanoflann.hpp>

struct MapPointCloudAdaptor {
    const std::vector<Point3D>& pts;

    inline size_t kdtree_get_point_count() const { return pts.size(); }

    inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
        return pts[idx].position[dim];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

// Explicitly use size_t as the 4th template parameter (IndexType)
using MapKDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, MapPointCloudAdaptor>,
    MapPointCloudAdaptor, 3, size_t>;

class MapManager {
public:
    explicit MapManager(int max_scans = 100);
    ~MapManager() = default;

    void addScan(const std::vector<Point3D>& scan_in_world);
    bool hasCloseNeighbor(const Eigen::Vector3d& query_pt, double r_max) const;
    void queryRadius(const Eigen::Vector3d& query_pt, double r, 
                     std::vector<Point3D>& out_neighbors) const;

    size_t totalPoints() const { return flat_map_points_.size(); }
    bool isEmpty() const { return flat_map_points_.empty(); }

private:
    int max_scans_;
    std::deque<std::vector<Point3D>> scan_buffer_;
    std::vector<Point3D> flat_map_points_;

    std::unique_ptr<MapPointCloudAdaptor> adaptor_;
    std::unique_ptr<MapKDTree> kd_tree_;

    void rebuildKDTree();
};

#endif // MAP_MANAGER_HPP