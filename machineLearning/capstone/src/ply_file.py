import pandas as pd

def write_ply_file2(out_file, num_vertices, cloud_points):
    """
    This function writes a .ply file with the point cloud points

    Parameters
    -------------------
    out_file: file
        file to write to

    num_vertics: int
        number of point cloud points

    cloud_points: np.array
        2D array of point cloud points. Each row is a point in the cloud

    Return Values
    -------------------
    None

    """
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
            line = str(point[0]) + ' ' + str(point[1]) + ' ' +  str(point[2]) + ' ' \
            +  str(point[3]) + '\n'
            out_file.write(line)

def write_ply_file(file_name, cloud_points):
    """
    This function writes a .ply file with the point cloud points

    Parameters
    -------------------
    file_name: string
        name of the file to output

    cloud_points: np.array
        2D array of point cloud points. Each row is a point in the cloud

    Return Values
    -------------------
    None

    """
    out_file = file(file_name, 'w')
    num_vertices = 0
    for point in cloud_points:
        if point is not None:
            num_vertices += 1

    write_ply_file2(out_file, num_vertices, cloud_points)

    out_file.close()

def read_ply_file(file_name):
    """
    This function reads in a .ply file. The ply file format follows this format:

    element vertex 16221
    property float x
    property float y
    property float z
    property float intensity
    end_header
    2.0 1.23099994659 0.922999978065 1.0
    2.04200005531 1.23800003529 0.93900001049 1.0
    2.00999999046 1.2009999752 0.92199999094 0.0
    2.03500008583 1.19799995422 0.930000007153 1.0
    ...

    The function skips over the header by looking for the end_header label. Then it 
    reads in the x,y,z,intensity values. The function assumes that is the format 
    of the file.

    Parameters
    -------------------
    file_name: string
        name of the ply file

    Return Values
    -------------------
    data: pandas.DataFrame
        data frame with the x,y,z,intensity values

    """
    # Find the end of the header
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

