#ifndef FEATURE_EXTRACTOR_HPP
#define FEATURE_EXTRACTOR_HPP

#include <vector>
#include <array>
#include <Eigen/Dense>

struct Point3D {
    Eigen::Vector3d position;
    Eigen::Vector3d normal;
    double a2d = 0.0; // Planarity factor
};

// Forward declaration of MapManager
class MapManager;

class FeatureExtractor {
public:
    explicit FeatureExtractor(int s_samples_per_list = 100, double r_outlier_max = 0.20);

    // 1. Compute per-point normal vector (via PCA) and planarity score a2D
    static void computeNormalsAndPlanarity(std::vector<Point3D>& scan_points, int k_neighbors = 10) ;

    // 2. Select top s samples from each of the 9 observability lists
    std::vector<Point3D> selectObservabilitySamples(const std::vector<Point3D>& scan_points,
                                                   const MapManager& map_manager) const;

private:
    int s_samples_;
    double r_max_;

    // Evaluates 9 observability scores for a point in the vehicle frame
    static std::array<double, 9> computeObservabilityScores(const Point3D& pt) ;
};

#endif // FEATURE_EXTRACTOR_HPP