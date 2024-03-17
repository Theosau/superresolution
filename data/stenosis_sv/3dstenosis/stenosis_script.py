'''
To generate steonses flows and predictions, we will need to:
    - Build a path.
    - Add contours at all the points of the path with the vessel radius.
    - Align the contours to prepare for lofting.
    - Loft the contours to generate the volume.
    - Create a solid from the loft.
    - Create a mesh from the solid.
    - Name the surfaces.
    - Set the boundary conditions.
    - Set the fluid parametes.
    - Solve the Navier-Stokes equations over the mesh and boundary conditions.
    - Transfer the mesh to image data.
    - Break down to 64x64x64.
    - Introduce in current algorithm.
'''

import sv, os, vtk, random, subprocess, math, argparse
# have to import vtk or else the get_polydata function fails!

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is my program description")
    parser.add_argument("--name", type=str, help="Name of the simulation.")
    parser.add_argument("--obs", type=float, help="Obstruction fraction, ranging 0--1.")
    parser.add_argument("--R", type=float, help="Base vessel radius.")
    parser.add_argument("--inflow", type=float, help="Inflow.")
    parser.add_argument("--ges", type=float, help="Global Edge size.")
    parser.add_argument("--refinement", type=float, help="Edge size refinement over stenosis region.")
    
    args = parser.parse_args()

    exmpl_name = args.name
    f0 = args.obs
    R = args.R
    inflow_val = args.inflow
    ges = args.ges
    refinement = args.refinement

    # Create new path object.
    path_name = "stenosis"
    path = sv.pathplanning.Path()

    # Give it some points.
    path.add_control_point([0.0, 0.0, 0.0])
    path.add_control_point([0.0, 0.0, 2.5])
    path.add_control_point([0.0, 0.0, 5.0])
    path.add_control_point([0.0, 0.0, 7.5])
    path.add_control_point([0.0, 0.0, 10.0])
    path.add_control_point([0.0, 0.0, 12.5])
    path.add_control_point([0.0, 0.0, 15.0])
    path.add_control_point([0.0, 0.0, 17.5])
    path.add_control_point([0.0, 0.0, 20.0])
    # path.add_control_point([0.0, 0.0, 25.0])
    # path.add_control_point([0.0, 0.0, 30.0])
    path_points = path.get_curve_points()

    # List of contour objects created.
    contours = []
    # List of polydata objects created from the contours.
    contours_plyds = []

    # base vessel radius
    # R = 2.0

    # f(x) = R * (1 - f0/2 *(1 + Cos(2.0*Pi * (x-x0)/L)))
    # stenosis parameters
    # f0 = 0.35 # obstruction fraction, ranging 0--1
    L = 10 # stenosis length
    z0 = 10 # maximum stenosis position

    for pid in range(len(path_points)):
        z = path_points[pid][-1]
        if z > 5 and z < 15:
            radius = R * (1 - f0/2 *(1 + math.cos(2.0*math.pi * (z-z0)/L)))
        else:
            radius = R

        # build contour around the point
        contour = sv.segmentation.Circle(
            radius = radius, 
            center = path_points[pid], 
            normal = path.get_curve_tangent(pid),
        )

        # add to the contours list and ploydata
        contours.append(contour)
        contours_plyds.append(contour.get_polydata())

    # Resample and align the contour polydata objects to ensure that all
    # contours contain the same quantity of points and are all rotated such that
    # the ids of each point in the contours are in the same position along the
    # contours for lofting.
    num_samples = 25    # Number of samples to take around circumference of contour.
    use_distance = True # Specify option for contour alignment.
    for index in range(len(contours_plyds)):
        # Resample the current contour.
        contours_plyds[index] = sv.geometry.interpolate_closed_curve(
            polydata=contours_plyds[index], 
            number_of_points=num_samples
        )

        # Align the current contour with the previous one for all contours but the first
        if not index is 0:
            contours_plyds[index] = sv.geometry.align_profile(
                contours_plyds[index - 1],
                contours_plyds[index],use_distance
            )

    # Loft the contours.
    # Set loft options.
    options = sv.geometry.LoftOptions()
    # Use linear interpolation between spline sample points.
    options.interpolate_spline_points = True
    # Set the number of points to sample a spline if
    # using linear interpolation between sample points.
    options.num_spline_points = 50
    # The number of longitudinal points used to sample splines.
    options.num_long_points = 200

    # Loft solid.
    lofted_surface = sv.geometry.loft(
        polydata_list=contours_plyds, 
        loft_options=options
    )

    # Create a new solid from the lofted solid.
    lofted_model = sv.modeling.PolyData()
    lofted_model.set_surface(surface=lofted_surface)

    # Cap the lofted volume.
    capped_model_pd = sv.vmtk.cap(
        surface=lofted_model.get_polydata(),
        use_center=False
    )

    # Import the capped model PolyData into model objects.
    capped_model = sv.modeling.PolyData()
    capped_model.set_surface(surface=capped_model_pd)

    # do the boundary faces thing here
    print(capped_model.identify_caps())

    # Export the solid to a polydata object written to /model/stenosis_model.vtp.
    os.mkdir(exmpl_name)
    os.chdir(exmpl_name)
    os.mkdir("model")
    capped_model.write(file_name=os.getcwd() + "/model/stenosis_model", format="vtp")


    ## Meshing
    mesher = sv.meshing.create_mesher(sv.meshing.Kernel.TETGEN)

    # base parameters
    global_edge_size = ges
    options = sv.meshing.TetGenOptions()
    options.global_edge_size = global_edge_size
    options.surface_mesh_flag = True
    options.volume_mesh_flag = True
    options.sphere_refinement_on = True
    # mesh refinment over the stenosis region
    options.sphere_refinement = [
        {
            "edge_size":global_edge_size/refinement,
            "radius":5,
            "center":[0, 0, 10],
        }
    ]
    # create the mesh
    mesher.set_model(capped_model)
    mesher.set_walls([1])
    mesher.generate_mesh(options)

    os.mkdir("mesh-complete")
    mesher.write_mesh(file_name=os.getcwd() + "/mesh-complete/mesh-complete.mesh.vtu")

    # Create a writer for .vtp file
    writer = vtk.vtkXMLPolyDataWriter()

    # mesh-complete.exterior.vtp
    surface_poly = mesher.get_surface()
    # Set the file name
    writer.SetFileName("mesh-complete/mesh-complete.exterior.vtp")
    # Set the input data
    writer.SetInputData(surface_poly)
    # Write the file
    writer.Write()

    # wall_combined.vtp
    wall_poly = mesher.get_face_polydata(1)
    # Set the file name
    writer.SetFileName("mesh-complete/walls_combined.vtp")
    # Set the input data
    writer.SetInputData(wall_poly)
    # Write the file
    writer.Write()


    ## Mesh Surfaces
    os.mkdir('mesh-complete/mesh-surfaces')
    # mesh-surfaces/wall.vtp
    wall_poly = mesher.get_face_polydata(1)
    # Set the file name
    writer.SetFileName("mesh-complete/mesh-surfaces/wall.vtp")
    # Set the input data
    writer.SetInputData(wall_poly)
    # Write the file
    writer.Write()

    # mesh-surfaces/inflow.vtp
    inflow_poly = mesher.get_face_polydata(2)
    # Set the file name
    writer.SetFileName("mesh-complete/mesh-surfaces/inflow.vtp")
    # Set the input data
    writer.SetInputData(inflow_poly)
    # Write the file
    writer.Write()

    # mesh-surfaces/outlet.vtp
    outflow_poly = mesher.get_face_polydata(3)
    # Set the file name
    writer.SetFileName("mesh-complete/mesh-surfaces/outlet.vtp")
    # Set the input data
    writer.SetInputData(outflow_poly)
    # Write the file
    writer.Write()


    ## Inflow boundary conditions
    filename = "inflow.flow"
    # inflow_val = 170
    # Open the file in write mode
    with open(filename, "w") as file:
        # Write text to the file
        text = "0.0 -{inflow_val}.\n".format(inflow_val=inflow_val)
        file.write(text)
        text = "1.0 -{inflow_val}.".format(inflow_val=inflow_val)
        file.write(text)
    print("File created successfully:", filename)

    ## Presolver file
    filename = "steady_manual.svpre"
    # Open the file in write mode
    with open(filename, "w") as file:
        # Write text to the file
        file.write("""mesh_and_adjncy_vtu mesh-complete/mesh-complete.mesh.vtu
    set_surface_id_vtp mesh-complete/mesh-complete.exterior.vtp 1
    set_surface_id_vtp mesh-complete/mesh-surfaces/inflow.vtp 2
    set_surface_id_vtp mesh-complete/mesh-surfaces/outlet.vtp 3
    fluid_density 1.06
    fluid_viscosity 0.04
    initial_pressure 0
    initial_velocity 0.0001 0.0001 0.0001
    prescribed_velocities_vtp mesh-complete/mesh-surfaces/inflow.vtp
    bct_analytical_shape parabolic
    bct_period 1.0
    bct_point_number 2
    bct_fourier_mode_number 1
    bct_create mesh-complete/mesh-surfaces/inflow.vtp inflow.flow
    bct_write_dat bct.dat
    bct_write_vtp bct.vtp
    pressure_vtp mesh-complete/mesh-surfaces/outlet.vtp 0
    noslip_vtp mesh-complete/walls_combined.vtp
    write_geombc geombc.dat.1
    write_restart restart.0.1""")
    print("File created successfully:", filename)

    ## Run the presolver
    command = "/Users/theophilesautory/Documents/BerkeleyPhD/Research/cfd/stenosis_sv/svSolver-build/svSolver-build/bin/svpre steady_manual.svpre"
    # Execute the command and wait for completion
    subprocess.run(command, shell=True, check=True)
    # Continue with the rest of your code
    print("Presolver ran successfully.")

    ## Numstart for simulation
    filename = "numstart.dat"
    # Open the file in write mode
    with open(filename, "w") as file:
        # Write text to the file
        file.write("0")
    print("File created successfully:", filename)