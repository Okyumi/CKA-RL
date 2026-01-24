import subprocess
import argparse
import random
from tasks import tasks

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algorithm",
        type=str,
        choices=[
            "simple",
            "componet",
            "finetune",
            "prognet",
            "packnet",
            "cka-rl",
            "masknet",
            "cbpnet",
            "crelus"
        ],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-run", default=False, action="store_true")
    parser.add_argument("--start-mode", type=int, default=0)
    parser.add_argument("--task-id", type=int, default=3)
    parser.add_argument("--tag", type=str, default="Debug")
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--capture-video", default=False, action="store_true")
    parser.add_argument("--video-every-n-episodes", type=int, default=50)
    parser.add_argument("--pool_size", default=20)
    parser.add_argument("--encoder_from_base", action="store_true")
    return parser.parse_args()


args = parse_args()

# modes = list(range(20)) if args.algorithm != "simple" else list(range(10))
modes = [args.task_id]
# modes = list(range(10)) 
# args.start_mode = 3
if args.algorithm not in ["simple", "packnet", "prognet", "cka-rl", "masknet", "cbpnet", "crelus"] and args.start_mode == 0:
    start_mode = 1
else:
    start_mode = args.start_mode

# Ensure start_mode exists in modes (important when modes is limited like [0])
if start_mode not in modes:
    start_mode = modes[0]  # Use first available task

run_name = (
    lambda task_id: f"task_{task_id}__{args.algorithm if task_id > 0 or args.algorithm in ['packnet', 'prognet', 'cka-rl', 'masknet', 'cbpnet', 'crelus'] else 'simple'}__run_contrastiveRL__{args.seed}"
)

first_idx = modes.index(start_mode)
for i, task_id in enumerate(modes[first_idx:]):
    params = f"--model-type={args.algorithm} --task-id={task_id} --seed={args.seed} --tag={args.tag}"
    if args.debug:
        params += " --total-timesteps=10000"
        params += " --learning_starts=5000"
    if args.encoder_from_base:
        params += " --encoder-from-base"
    else:
        params += " --no-encoder-from-base"
    if args.track:
        params += " --track"
    if args.capture_video:
        params += " --capture-video"
    params += f" --video-every-n-episodes={args.video_every_n_episodes}"
    
    save_dir = f"agents/{args.tag}"
    params += f" --save-dir={save_dir}"
    params += f" --pool_size={args.pool_size}"

    if first_idx > 0 or i > 0:
        # multiple previous modules
        if args.algorithm in ["componet", "prognet", "cka-rl"]:
            params += " --prev-units"
            for i in modes[: modes.index(task_id)]:
                params += f" {save_dir}/{run_name(i)}"
        # single previous module
        elif args.algorithm in ["finetune", "packnet", "masknet", "cbpnet", "crelus"]:
            params += f" --prev-units {save_dir}/{run_name(task_id-1)}"

    # Launch experiment
    cmd = f"python3 run_contrastiveRL.py {params}"
    print(cmd)

    if not args.no_run:
        res = subprocess.run(cmd.split(" "))
        if res.returncode != 0:
            print(f"*** Process returned code {res.returncode}. Stopping on error.")
            quit(1)
