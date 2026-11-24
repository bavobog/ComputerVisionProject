#include "code/FeatureExtractor.h"
#include "code/MapManager.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cassert>

// Helper to generate a synthetic corridor (ground plane + left/right walls)
std::vector<Point3D> generateCorridorScan(double y_offset) {
    std::vector<Point3D> scan;

    // 1. Ground Plane (Z = 0, X in [-4, 4], Y in [y_offset, y_offset + 10])
    for (double x = -4.0; x <= 4.0; x += 0.5) {
        for (double y = y_offset; y <= y_offset + 10.0; y += 0.5) {
            Point3D pt;
            pt.position = Eigen::Vector3d(x, y, 0.0);
            scan.push_back(pt);
        }
    }

    // 2. Left Wall (X = -4, Y in [y_offset, y_offset + 10], Z in [0, 3])
    for (double y = y_offset; y <= y_offset + 10.0; y += 0.5) {
        for (double z = 0.5; z <= 3.0; z += 0.5) {
            Point3D pt;
            pt.position = Eigen::Vector3d(-4.0, y, z);
            scan.push_back(pt);
        }
    }

    // 3. Right Wall (X = 4, Y in [y_offset, y_offset + 10], Z in [0, 3])
    for (double y = y_offset; y <= y_offset + 10.0; y += 0.5) {
        for (double z = 0.5; z <= 3.0; z += 0.5) {
            Point3D pt;
            pt.position = Eigen::Vector3d(4.0, y, z);
            scan.push_back(pt);
        }
    }

    return scan;
}

int main() {
    std::cout << "===================================================\n";
    std::cout << "   Testing Task B: Observability & Map Management  \n";
    std::cout << "===================================================\n";

    FeatureExtractor extractor(100 /* s samples per list */, 0.20 /* r_max */);
    MapManager map_manager(100 /* max scans in window */);

    // -------------------------------------------------------------
    // Test Step 1: Scan 1 Processing & Initial Map Population
    // -------------------------------------------------------------
    std::cout << "\n[1] Processing Scan 1 (At origin Y = 0.0)..." << std::endl;
    auto scan1 = generateCorridorScan(0.0);
    std::cout << "    Generated Scan 1 with " << scan1.size() << " points." << std::endl;

    extractor.computeNormalsAndPlanarity(scan1, 10);
    std::cout << "    Computed PCA Normals and a2D for Scan 1." << std::endl;

    // Simulate scan 1 localized at origin and added to map
    map_manager.addScan(scan1);
    std::cout << "    Added Scan 1 to MapManager. Map point count: " << map_manager.totalPoints() << std::endl;
    assert(map_manager.totalPoints() == scan1.size());

    // -------------------------------------------------------------
    // Test Step 2: Scan 2 Processing & Observability Sampling
    // -------------------------------------------------------------
    std::cout << "\n[2] Processing Scan 2 (Vehicle moved to Y = 0.5)..." << std::endl;
    auto scan2 = generateCorridorScan(0.5);
    extractor.computeNormalsAndPlanarity(scan2, 10);

    // Select observability samples (up to 9 * 100 = 900 points) against current map
    std::vector<Point3D> sampled_points = extractor.selectObservabilitySamples(scan2, map_manager);
    std::cout << "    Selected " << sampled_points.size() << " observability samples out of "
              << scan2.size() << " points." << std::endl;

    // Verify sampling bounds (should be <= 9 * s)
    assert(sampled_points.size() > 0 && sampled_points.size() <= 900);

    // -------------------------------------------------------------
    // Test Step 3: Verify Map Manager Radius Search (For Person C)
    // -------------------------------------------------------------
    std::cout << "\n[3] Testing IMLS Radius Query Interface for Person C..." << std::endl;
    Eigen::Vector3d test_query(0.0, 2.0, 0.0); // Point on ground
    std::vector<Point3D> neighbors;
    double query_radius = 0.20; // r = 0.20 m

    map_manager.queryRadius(test_query, query_radius, neighbors);
    std::cout << "    Queried points within r = " << query_radius << " m around [0, 2, 0]." << std::endl;
    std::cout << "    Found " << neighbors.size() << " map neighbors." << std::endl;

    assert(!neighbors.empty());
    std::cout << "    Sample Neighbor Normal: [" << neighbors[0].normal.transpose()
              << "] | a2D: " << std::fixed << std::setprecision(4) << neighbors[0].a2d << std::endl;

    // -------------------------------------------------------------
    // Test Step 4: Verify Outlier Distance Check
    // -------------------------------------------------------------
    std::cout << "\n[4] Testing Outlier Rejection Check..." << std::endl;
    Eigen::Vector3d inlier_pt(0.0, 2.0, 0.0);
    Eigen::Vector3d outlier_pt(50.0, 50.0, 50.0); // Way outside map

    bool inlier_result = map_manager.hasCloseNeighbor(inlier_pt, 0.20);
    bool outlier_result = map_manager.hasCloseNeighbor(outlier_pt, 0.20);

    std::cout << "    Inlier check [0, 2, 0] (Expect true): " << (inlier_result ? "PASS" : "FAIL") << std::endl;
    std::cout << "    Outlier check [50, 50, 50] (Expect false): " << (!outlier_result ? "PASS" : "FAIL") << std::endl;

    assert(inlier_result == true);
    assert(outlier_result == false);

    std::cout << "\n===================================================\n";
    std::cout << " >>> ALL TASK B MODULE TESTS PASSED SUCCESSFULLY! <<< \n";
    std::cout << "===================================================\n";

    return 0;
}