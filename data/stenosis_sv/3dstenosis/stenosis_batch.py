import subprocess

if __name__ == "__main__":
    for obs in [0.32, 0.33, 0.34, 0.36, 0.37, 0.38, 0.39]:
        if obs > 0.5:
            refinement = 4
        elif obs > 0.3:
            refinement = 3
        else:
            refinement = 2
        for inflow in [40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0]:
            command = "simvascular --python -- simulate_stenosis.py \
                --obs {obs} \
                --R 3.0 \
                --inflow {inflow} \
                --ges 0.4 \
                --refinement {refinement} \
                --tss 0.010 \
                --nt 400".format(
                    obs = obs, 
                    inflow=inflow,
                    refinement = refinement,
                )

            try:
                # Execute the command and wait for completion
                subprocess.run(command, shell=True, check=True)
                print("DONE WITH INFLOW {inflow}".format(inflow=inflow))
            except subprocess.CalledProcessError:
                print("Failed for {inflow}".format(inflow=inflow))       