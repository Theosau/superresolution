import subprocess

if __name__ == "__main__":
    num = 0
    for obs in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50]: #21
    # for obs in [1.5]:
        refinement = 1.5
        for inflow in [35.0, 37.5, 40.0, 42.5, 45.0, 47.5, 50.0, 52.5, 55.0, 57.5, 60.0, 62.5, 65.0, 67.5, 70.0, 72.5, 75.0]: #17
        # for inflow in [40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0]:
        # for inflow in [75.0]:
            command = "simvascular --python -- simulate_aneurysm.py \
                --obs {obs} \
                --R 2.0 \
                --inflow {inflow} \
                --ges 0.3 \
                --refinement {refinement} \
                --tss 0.010 \
                --nt 100".format(
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
            # print counter just to know
            
            num += 1
            print(num, "/357")