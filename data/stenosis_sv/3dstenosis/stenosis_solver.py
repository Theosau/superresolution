import subprocess, os, argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is my program description")
    parser.add_argument("--name", type=str, help="Name of the simulation.")
    parser.add_argument("--tss", type=float, help="Time Step Size.")
    parser.add_argument("--nt", type=float, help="Number of Timesteps.")
    
    args = parser.parse_args()

    exmpl_name = args.name
    tss = args.tss
    nt = args.nt

    # exmpl_name = "example6"
    os.chdir(exmpl_name)
    ## SOlver file
    filename = "solver.inp"
    # Open the file in write mode
    with open(filename, "w") as file:
        # Write text to the file
        file.write("""Density: 1.06
    Viscosity: 0.04
    """)
        file.write("Number of Timesteps: {nt} \n".format(nt=nt))
        file.write("Time Step Size: {tts} \n".format(tts=tss))
        file.write("""
    Number of Timesteps between Restarts: 100
    Number of Force Surfaces: 1
    Surface ID's for Force Calculation: 1
    Force Calculation Method: Velocity Based
    Print Average Solution: True
    Print Error Indicators: False

    Time Varying Boundary Conditions From File: True

    Step Construction: 0 1 0 1

    Number of Resistance Surfaces: 1
    List of Resistance Surfaces: 3
    Resistance Values: 1333

    Pressure Coupling: Implicit
    Number of Coupled Surfaces: 1

    Backflow Stabilization Coefficient: 0.2
    Residual Control: True
    Residual Criteria: 0.01
    Minimum Required Iterations: 3
    svLS Type: NS
    Equation of State: Incompressible
    Number of Krylov Vectors per GMRES Sweep: 100
    Number of Solves per Left-hand-side Formation: 1
    Tolerance on Momentum Equations: 0.05
    Tolerance on Continuity Equations: 0.4
    Tolerance on svLS NS Solver: 0.4
    Maximum Number of Iterations for svLS NS Solver: 1
    Maximum Number of Iterations for svLS Momentum Loop: 2
    Maximum Number of Iterations for svLS Continuity Loop: 400
    Time Integration Rule: Second Order
    Time Integration Rho Infinity: 0.5
    Flow Advection Form: Convective
    Quadrature Rule on Interior: 2
    Quadrature Rule on Boundary: 3  
        
    """)
    print("File created successfully:", filename)

    # Run the simulation
    command = "mpiexec -np 4 /Users/theophilesautory/Documents/BerkeleyPhD/Research/cfd/stenosis_sv/svSolver-build/svSolver-build/bin/svsolver solver.inp"
    # Execute the command and wait for completion
    subprocess.run(command, shell=True, check=True)
    # Continue with the rest of your code
    print("Solver ran successfully.")