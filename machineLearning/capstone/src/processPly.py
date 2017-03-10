import pandas as pd
import numpy as np
import time
import pickle
from scipy import spatial
from scipy import stats
from pandas.tools.plotting import scatter_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

from os import listdir
from os.path import isfile, join

def write_poly_file(out_file, num_vertices, cloud_points):
    out_file.write("ply\n")
    out_file.write("format ascii 1.0\n")
    vertex_string = str()
    vertex_string = "element vertex " + str(num_vertices) + "\n"
    out_file.write(vertex_string)
    out_file.write("property float x\n")
    out_file.write("property float y\n")
    out_file.write("property float z\n")
    out_file.write("property float intensity\n")
    out_file.write("end_header\n")

    for point in cloud_points:
        if point is not None:
            line = str()
            line = str(point[0]) + ' ' + str(point[1]) + ' ' +  str(point[2]) + ' ' +  str(point[3]) + '\n'
            out_file.write(line)

def write_poly_file2(out_file, cloud_points):

    num_vertices = 0
    for point in cloud_points:
        if point is not None:
            num_vertices += 1

    write_poly_file(out_file, num_vertices, cloud_points)

def load_class_id(cordon_points_file_name, num_point_cloud_points, kdtree):
    data = read_ply_file(cordon_points_file_name)

    cordon_points = np.array(data[['x','y','z']])

    class_id = np.zeros(num_point_cloud_points, dtype=np.int)

    for query_point in cordon_points:
        point_distance, point_index = kdtree.query(query_point, k=1, p=np.inf)
        class_id[point_index] = 1
    
    return class_id

def load_training_data(point_cloud_file_name, cordon_points_file_name, cordon2_points_file_name=None):
    data = read_ply_file(point_cloud_file_name)

    point_cloud_points = np.array(data[['x','y','z']])
    point_cloud_intensities = np.array(data['intensity'])

    # Build a KD tree to make finding nearest neighbors faster. Use only x,y,z points (not intensity) to build tree
    kdtree = spatial.cKDTree(point_cloud_points)

    data = calculate_features(point_cloud_points, point_cloud_intensities, kdtree)
     
    class_id = load_class_id(cordon_points_file_name, cordon2_points_file_name, point_cloud_points.shape[0], kdtree)
    data['class_id'] = pd.Series(class_id, index=data.index)

    return data

def calculate_features_old(point_cloud_points, point_cloud_intensities, kdtree, pca_min_radius = 0.06):
    # For each point in the cloud, calculate a covariance
    pca_max_points = 200
    pca_min_points = 15
    cntr = 0

    feature_names = ["x", "y", "z", "intensity", "sxx", "sxy", "sxz", "syy", "syz", "szz", "eigVal1", "eigVal2", "eigVal3", "maxDist", "meanDist", "varDist"]
    features = np.empty([point_cloud_points.shape[0], len(feature_names)])

    feature_index = 0
    for query_point in point_cloud_points:
        points_distance, points_index = kdtree.query(query_point, k=pca_max_points, distance_upper_bound=pca_min_radius, p=np.inf)
        # if we don't get enough points to do pca, expand the radius to ensure you get a minimum number of points
        num_points = np.count_nonzero(np.isfinite(points_distance))
        if (num_points  < pca_min_points):
            # print "too few points ", num_points
            points_distance, points_index = kdtree.query(query_point, k=pca_min_points)
        # Get a list of the points (with dist < inf)
        points = [point_cloud_points[index] for dist, index in zip(points_distance, points_index) if np.isfinite(dist)]
        points = np.transpose(np.array(points))
        # points is now a matrix with columns of (x,y,z) triplets

        # Find the covariance of the points_index
        cov = np.cov(points)

        # Find the eigen values of the covariance matrix
        w = np.linalg.eigvalsh(cov)

        # Get a list of the point cloud intensities (with dist < inf)
        intensities = [point_cloud_intensities[index] for dist, index in zip(points_distance, points_index) if np.isfinite(dist)]
        intensities = np.transpose(np.array(intensities))
        # Get stats on intensity
        [num, minmax_intensity, mean_intensity, var_intensity, skew_intensity, kurt_intensity] = stats.describe(intensities)

        # Find the stats on mean point distance
        distances = [dist for dist in points_distance if np.isfinite(dist)]
        [num, minmax_distance, mean_distance, var_distance, skew_distance, kurt_distance] = stats.describe(distances)

        new_feature = [points[0][0], points[1][0], points[2][0], intensities[0], cov[0,0], cov[0,1], cov[0,2], cov[1,1], cov[1,2], cov[2,2], w[0], w[1], w[2], 
            minmax_distance[1], mean_distance, var_distance]
        
        # print "new_feature = ", new_feature
        features[feature_index,:] = new_feature
        feature_index += 1

    data = pd.DataFrame(data=features, columns=feature_names)
    return data

def calculate_features(point_cloud_points, point_cloud_intensities, kdtree, pca_min_radius=0.06):
    # For each point in the cloud, calculate a covariance
    pca_max_points = 200
    pca_min_points = 15
    cntr = 0

    feature_names = ["x", "y", "z", "intensity", "pointness", "surfaceness", "linearness",
                     "cos_tangent", "sin_tangent", "cos_normal", "sin_normal" ]
    features = np.empty([point_cloud_points.shape[0], len(feature_names)])

    feature_index = 0
    for query_point in point_cloud_points:
        points_distance, points_index = kdtree.query(query_point, k=pca_max_points,
                                                     distance_upper_bound=pca_min_radius, p=np.inf)
        # if we don't get enough points to do pca, expand the radius to
        # ensure you get a minimum number of points
        num_points = np.count_nonzero(np.isfinite(points_distance))
        if num_points < pca_min_points:
            # print "too few points ", num_points
            points_distance, points_index = kdtree.query(query_point, k=pca_min_points)
        # Get a list of the points (with dist < inf)
        points = [point_cloud_points[index]
                  for dist, index in zip(points_distance, points_index) if np.isfinite(dist)]
        points = np.transpose(np.array(points))
        # points is now a matrix with columns of (x,y,z) triplets

        # Find the covariance of the points_index
        cov = np.cov(points)

        # Find the eigen values(ascending), vectors of the covariance matrix
        w, v = np.linalg.eigh(cov)

        # Spectral features from Munoz icra 2009
        pointness = w[0]
        surfaceness = w[1] - w[0]
        linearness = w[2] - w[1]
        max_spectral_feature = max([pointness, surfaceness, linearness])
        # Directional Features from Munoz icra 2009
        tangent = v[2]
        normal = v[0]

        # Find the sine and cosine of the tangent line w.r.t. horizontal (x,y) plane
        adjacent = np.sqrt(tangent[0]**2 + tangent[1]**2)
        opposite = tangent[2]
        hypotenuse = np.sqrt(adjacent**2 + opposite**2)
        cos_tangent = adjacent / hypotenuse
        sin_tangent = opposite / hypotenuse
        # scale the values based on strength of the extracted directions
        scale = linearness / max_spectral_feature
        cos_tangent *= scale
        sin_tangent *= scale
        # Find the sine and cosine of the normal line w.r.t. horizontal (x,y) plane
        adjacent = np.sqrt(normal[0]**2 + normal[1]**2)
        opposite = normal[2]
        hypotenuse = np.sqrt(adjacent**2 + opposite**2)
        cos_normal = adjacent / hypotenuse
        sin_normal = opposite / hypotenuse
        # scale the values based on strength of the extracted directions
        scale = surfaceness / max_spectral_feature
        cos_normal *= scale
        sin_normal *= scale

        new_feature = [points[0][0], points[1][0], points[2][0],
                       point_cloud_intensities[points_index[0]],
                       pointness, surfaceness, linearness, cos_tangent,
                       sin_tangent, cos_normal, sin_normal]

        # print "new_feature = ", new_feature
        features[feature_index, :] = new_feature
        feature_index += 1

    data = pd.DataFrame(data=features, columns=feature_names)
    return data

def read_ply_file(file_name):
    # Find the end of the header
    done = False
    skip_rows = 0
    data = pd.DataFrame()
    with open(file_name) as f:
        # skip the header information
        for line in f:
            if line[0:len('end_header')] == 'end_header':
                skip_rows += 1
                break
            else:
                skip_rows += 1
        # read in the x,y,z, intensity values
        for line in f:
            nums = [float(i) for i in line.split()]
            new_data = pd.DataFrame([nums], columns=['x', 'y', 'z', 'intensity'])
            data = data.append(new_data, ignore_index=True)
    return data

def calculate_point_cloud_features(file_name, pca_min_radius=0.06):
    data = read_ply_file(file_name)

    point_cloud_points = np.array(data[['x', 'y', 'z']])
    point_cloud_intensities = np.array(data['intensity'])

    # Build a KD tree to make finding nearest neighbors faster. 
    # Use only x,y,z points (not intensity) to build tree
    kdtree = spatial.cKDTree(point_cloud_points)

    data = calculate_features(point_cloud_points, point_cloud_intensities, kdtree,
                              pca_min_radius=pca_min_radius)
    return data

def animate_classifier(data_file_path, classifier_save_file_name):
    clf = pickle.load(open(classifier_save_file_name, "rb"))

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)

    fig = plt.figure()
    cane_plt, = plt.plot([], [], 'r.')
    cordon_plt, = plt.plot([], [], 'b.')
    plt.xlim(-1.0, 1.0)
    plt.ylim(-1.0, 1.0)
    file_text = plt.text(-0.5, -0.875, '')

    with writer.saving(fig, "writer_test.mp4", 100):
        only_files = [f for f in listdir(data_file_path) if isfile(join(data_file_path, f))]
        ply_files = [f for f in only_files if f[-9:] == '.flt4.ply']
        ply_files.sort()
        # ply_files = [ply_files[index] for index in range(30)]

        for point_cloud_file_name in ply_files:
            point_cloud_extension = '.flt4.ply'
            # Only animate the point cloud files. Do not animate the cordon point files, etc.
            if point_cloud_file_name[-len(point_cloud_extension):] == point_cloud_extension:
                data = calculate_point_cloud_features(data_file_path+point_cloud_file_name)
                # Remove the point cloud x,y,z data from the data frame. Those will not be used as features
                point_cloud_points = data[['x','y','z','intensity']]

                del data['x']
                del data['y']
                del data['z']
                del data['intensity']
                
                X_all = data
                predicted_class = clf.predict(X_all)
                
                classified_point_cloud = point_cloud_points[['x','y','z']]
                classified_point_cloud['class_id'] = predicted_class

                cordon_points = classified_point_cloud.loc[classified_point_cloud['class_id']== 1]
                cane_points = classified_point_cloud.loc[classified_point_cloud['class_id'] == 0]

                cordon_plt.set_data(-cordon_points['z'], -cordon_points['y'])
                cane_plt.set_data(-cane_points['z'], -cane_points['y'])
                file_text.set_text(point_cloud_file_name)
                writer.grab_frame() 

def test_calculate_point_cloud_features():
    # Data file from rosbag play 2017-01-18-13-50-16.bag
    file_name = "/media/sroth/adata/gallo/cordonTracking/data/ply/1484765416981987000.flt4.ply"
    tstart = time.time()
    data = calculate_point_cloud_features(file_name)
    print ("execution time = ", time.time() - tstart)
    print data.head
    # scatter_matrix(data['eigVal1', 'eigVal2', 'eigVal3'])

def test_load_training_data():
    point_cloud_file_name = "/media/sroth/adata/gallo/cordonTracking/data/ply/1484765416981987000.flt4.ply"
    cordon_points_file_name = "/media/sroth/adata/gallo/cordonTracking/data/ply/1484765416981987000-Cordon.ply"
    data = load_training_data(point_cloud_file_name, cordon_points_file_name)
    print data

#test_load_training_data()
#animate_classifier("/media/sroth/adata/gallo/cordonTracking/data/ply/", "/media/sroth/adata/gallo/cordonTracking/src/classifier.p")