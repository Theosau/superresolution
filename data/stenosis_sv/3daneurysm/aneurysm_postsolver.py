import os, subprocess, argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is my program description")
    parser.add_argument("--name", type=str, help="Name of the simulation.")
    parser.add_argument("--nt", type=float, help="Number of Timesteps.")
    
    args = parser.parse_args()

    exmpl_name = args.name
    nt = args.nt

    ## Prepare simulation output directory
    # exmpl_name = "example6"
    os.chdir(exmpl_name)
    os.mkdir('4-procs_case/steady-converted-results')

    # Run the post simulation
    command = "/Users/theophilesautory/Documents/BerkeleyPhD/Research/cfd/stenosis_sv/svSolver-build/svSolver-build/bin/svpost \
        -all \
        -indir 4-procs_case \
        -outdir 4-procs_case/steady-converted-results \
        -start {nt} \
        -stop {nt} \
        -inc 100 \
        -vtp all_results \
        -vtu all_results".format(nt=nt)
    # Execute the command and wait for completion
    subprocess.run(command, shell=True, check=True)
    # Continue with the rest of your code
    print("Post solver ran successfully.")