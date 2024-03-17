import argparse, subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is my program description")
    parser.add_argument("--obs", type=float, help="Obstruction fraction, ranging 0--1.")
    parser.add_argument("--R", type=float, help="Base vessel radius.")
    parser.add_argument("--inflow", type=float, help="Inflow.")
    parser.add_argument("--ges", type=float, help="Global Edge size.")
    parser.add_argument("--refinement", type=float, help="Edge size refinement over stenosis region.")
    parser.add_argument("--tss", type=float, help="Time Step Size.")
    parser.add_argument("--nt", type=float, help="Number of Timesteps.")
    
    args = parser.parse_args()

    obs = args.obs
    R = args.R
    inflow = args.inflow
    ges = args.ges
    refinement = args.refinement
    tss = args.tss
    nt = args.nt

    name = "ex_{obs}_{R}_{inflow}_{ges}_{refinement}_{tss}_{nt}".format(
        obs = obs,
        R = R,
        inflow = inflow,
        ges = ges,
        refinement = refinement,
        tss = tss,
        nt = nt,
    ).replace(".", "")
    
    # Run the set-up 
    command = "/usr/local/bin/simvascular --python -- stenosis_script.py \
        --name {name} \
        --obs {obs} \
        --R {R} \
        --inflow {inflow} \
        --ges {ges} \
        --refinement {refinement}".format(
            name = name,
            obs = obs,
            R = R,
            inflow = inflow,
            ges = ges,
            refinement = refinement,
        )
    # Execute the command and wait for completion
    subprocess.run(command, shell=True, check=True)

    # Run the simulation
    command = "/usr/local/bin/simvascular --python -- stenosis_solver.py \
        --name {name} \
        --tss {tss} \
        --nt {nt}".format(
            name = name,
            tss = tss,
            nt = nt,
        )
    # Execute the command and wait for completion
    subprocess.run(command, shell=True, check=True)

    # Run the post solver
    command = "/usr/local/bin/simvascular --python -- stenosis_postsolver.py \
        --name {name} \
        --nt {nt}".format(
            name = name,
            nt = nt,
        )
    # Execute the command and wait for completion
    subprocess.run(command, shell=True, check=True)

    print('SIMULATION AND POST-PROCESSING DONE.')

