import argparse
import os
import subprocess

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run noisy BARF experiments")
    parser.add_argument(
        "--data_dir", type=str, default="data", help="UltraNeRF dataset directory"
    )
    args = parser.parse_args()

    # Set noisy position config
    tss = [0.0, 0.01, 0.05]
    rss = [0.0, 0.1, 0.5]
    perturb_ratios = [0.1, 0.5]
    repeats = 1

    # Create configs for each combination except ts and rs both 0.0
    configs = []
    for ts in tss:
        for rs in rss:
            if ts == 0.0 and rs == 0.0:
                continue
            for pr in perturb_ratios:
                for _ in range(repeats):
                    configs.append((ts, rs, pr))

    for ts, rs, pr in configs:
        rotation_strength = rs
        translation_strength = ts

        print(
            f"Rotation Strength: {rotation_strength}, Translation Strength: {translation_strength}, Perturb Ratio: {pr}"
        )

        for i in range(repeats):

            expname = f"noisy_barf_only_pose_{rotation_strength}_{translation_strength}_{pr}_{i}"
            pose_path = os.path.join(
                args.data_dir,
                "noisy_poses",
                f"{rotation_strength}_{translation_strength}_{pr}_{i}.npy",
            )

            # Construct the training command
            train_command = [
                "python",
                "run_barf.py",
                "--expname",
                expname,
                "--pose_path",
                pose_path,
                "--config",
                "config_base_barf.txt",
                "--tensorboard",
                # "--n_iters", "50",
                # "--i_print", "10",
                # "--i_weights", "10"
                "--n_iters",
                "100000",
                "--i_print",
                "2000",
                "--i_weights",
                "50000",
            ]

            # Print and execute the command
            print(f"Running command for BARF: {' '.join(train_command)}")
            try:
                subprocess.run(train_command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Command failed with return code {e.returncode}")

            expname = f"noisy_nerf_only_pose_{rotation_strength}_{translation_strength}_{pr}_{i}"

            train_command = [
                "python",
                "run_ultranerf.py",
                "--expname",
                expname,
                "--pose_path",
                pose_path,
                "--config",
                "config_base_nerf.txt",
                "--tensorboard",
                # "--n_iters", "50",
                # "--i_print", "10",
                # "--i_weights", "10"
                "--n_iters",
                "100000",
                "--i_print",
                "2000",
                "--i_weights",
                "50000",
            ]

            # Print and execute the command
            print(f"Running command for Ultranerf: {' '.join(train_command)}")
            try:
                subprocess.run(train_command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Command failed with return code {e.returncode}")
