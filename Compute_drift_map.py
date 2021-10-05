from pypointmatcher import pointmatcher as pm, pointmatchersupport as pms
from pathlib import Path
import numpy as np
import sys

if(len(sys.argv)!=5):
    print("Error in the number of arguments. "
          "1: path of the reference point cloud file, "
          "2: input directory name of the reading point-cloud, "
          "3: output directory name for the results, "
          "4: file name of the resulted .vtk")
    print('Exemple: python3 Compute_drift_map.py "/home/reference.vtk" "/home/map/" "/home/drift/" "map_drift""')
    sys.exit()

reference_path = sys.argv[1]
input_directory = sys.argv[2]
output_base_directory = sys.argv[3]
output_base_file = sys.argv[4]

PM = pm.PointMatcher
DP = PM.DataPoints
Parameters = pms.Parametrizable.Parameters

# Load 3D point clouds
ref = DP(DP.load(reference_path))
txt_folder = Path(input_directory).rglob('map_*')  #input files should for instance map_1.vtk
files = [x for x in txt_folder]
files_sorted = sorted(files, key=lambda x: int((str(x).split('/'))[-1].split('_')[1].split('.')[0]))

for i in files_sorted:
    number_file = int((str(i).split('/'))[-1].split('_')[1].split('.')[0])
    data = DP(DP.load(str(i)))
    icp = PM.ICP()
    params = Parameters()
    pms.setLogger(PM.get().LoggerRegistrar.create("FileLogger"))
    # One knn to have one neighbourhood for each point of the reading
    params = Parameters()
    params["knn"] = "1"  # for Hausdorff distance, we only need the first closest point
    params["epsilon"] = "0"
    matcher_Hausdorff = PM.get().MatcherRegistrar.create("KDTreeMatcher", params)
    matcher_Hausdorff.init(ref)
    matches = matcher_Hausdorff.findClosests(data)
    outlier_weights = icp.outlierFilters.compute(data, ref, matches)
    matched_points = PM.ErrorMinimizer.ErrorElements(data, ref, outlier_weights, matches)
    dim = matched_points.reading.getEuclideanDim()
    nb_matched_points = matched_points.reading.getNbPoints()
    matched_read = matched_points.reading.features[:dim]
    matched_ref = matched_points.reference.features[:dim]
    
    result = data.createSimilarEmpty()
    result.features=matched_points.reading.features[:dim]
    distance_drift = np.linalg.norm(matched_read-matched_ref, axis=0)
    result.addDescriptor("Drift", np.array([distance_drift]))
    result.save(f"{output_base_directory}{output_base_file}_{number_file}.vtk")

