# contrastive RL (sigmoid NCE) training script
import os
import glob
import random
import time
from dataclasses import dataclass
from tqdm import tqdm

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import pathlib
from torch.utils.tensorboard import SummaryWriter
from typing import Literal, Optional, Tuple
import wandb
from models import SimpleAgent, CompoNetAgent, PackNetAgent, ProgressiveNetAgent, CkaRlAgent, MaskNetAgent, CbpAgent, CReLUsAgent
from tasks import get_task
from utils.AdamGnT import AdamGnT
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
from models.cbp_modules import GnT
from wrapper.goal_wrapper import GoalObsWrapper


@dataclass
class Args:
    model_type: Literal["simple", "finetune", "componet", "packnet", "prognet", "cka-rl", "masknet", "cbpnet", "crelus"]
    """The name of the NN model to use for the agent"""
    save_dir: Optional[str] = None
    """If provided, the model will be saved in the given directory"""
    prev_units: Tuple[pathlib.Path, ...] = ()
    """Paths to the previous models. Not required when model_type is `simple` or `packnet` or `prognet`"""

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "metaworld-single-sac"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    video_every_n_episodes: int = 50
    """record one episode every N episodes"""
    video_dir: str = "videos"
    """base directory for recorded videos"""

    # Algorithm specific arguments
    task_id: int = 0
    """ID number of the task"""
    eval_every: int = 10_000
    """Evaluate the agent in determinstic mode every X timesteps"""
    num_evals: int = 10
    """Number of times to evaluate the agent"""
    total_timesteps: int = int(1e6)
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5_000
    """timestep to start learning"""
    random_actions_end: int = 10_000
    """timesteps to take actions randomly"""
    policy_lr: float = 1e-3
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the contrastive encoders"""
    tag: str = "Debug"
    """experiment tag"""
    pool_size: int = 9
    """pool size"""
    encoder_from_base: bool = False
    """load encoder from base_dir"""
    # The following parameters are for the contrastive loss
    nce_loss_weight: float = 1.0
    """weight for the InfoNCE loss"""
    nce_temperature: float = 0.1
    """temperature for InfoNCE logits"""
    nce_proj_dim: int = 128
    """projection dimension for contrastive features"""
    nce_hidden_dim: int = 256
    """hidden dimension for contrastive encoder"""
    nce_update_freq: int = 1
    """update contrastive loss every N steps"""
    nce_start: int = 5_000
    """global step to start contrastive updates"""
    debug_print_interval: int = 1
    """print state/action/goal samples every N steps (0 disables)"""

def make_env(task_id, capture_video, run_name, video_every_n_episodes, video_dir):
    def thunk():
        env = get_task(task_id)
        if capture_video:
            env.render_mode = "rgb_array"
        env = GoalObsWrapper(env, goal_dim=3)
        if capture_video:
            video_path = os.path.join(video_dir, run_name)
            env = gym.wrappers.RecordVideo(
                env,
                video_path,
                episode_trigger=lambda ep: ep % video_every_n_episodes == 0,
            )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def get_state_goal_dims(obs_space):
    if isinstance(obs_space, gym.spaces.Dict):
        state_space = obs_space.get("observation", obs_space.get("state", None))
        goal_space = obs_space.get("desired_goal", obs_space.get("goal", None))
        critic_goal_space = obs_space.get("critic_goal", None)
        if state_space is None:
            state_dim = int(
                sum(np.prod(space.shape) for space in obs_space.spaces.values())
            )
        else:
            state_dim = int(np.prod(state_space.shape))
        goal_dim = int(np.prod(goal_space.shape)) if goal_space is not None else 0
        critic_goal_dim = (
            int(np.prod(critic_goal_space.shape)) if critic_goal_space is not None else 0
        )
        return state_dim, goal_dim, critic_goal_dim
    state_dim = int(np.prod(obs_space.shape))
    # Placeholder: use a dummy goal with same dim as state
    return state_dim, state_dim, state_dim


def _concat_last_dim(parts):
    if isinstance(parts[0], torch.Tensor):
        return torch.cat(parts, dim=-1)
    return np.concatenate(parts, axis=-1)


def flatten_dict_obs(obs_dict):
    preferred_keys = [
        "observation",
        "state",
        "achieved_goal",
        "desired_goal",
        "goal",
    ]
    keys = [key for key in preferred_keys if key in obs_dict]
    if not keys:
        keys = sorted(obs_dict.keys())
    parts = [obs_dict[key] for key in keys]
    return _concat_last_dim(parts)


def split_obs_and_goal(obs):
    """Extract state, actor goal, and critic goal from wrapped env outputs."""
    if isinstance(obs, dict):
        state = obs.get("observation", obs.get("state", None))
        actor_goal = obs.get("desired_goal", obs.get("goal", None))
        critic_goal = obs.get("critic_goal", None)
        if state is None:
            state = flatten_dict_obs(obs)
        return state, actor_goal, critic_goal
    return obs, None, None


def build_policy_input(state, actor_goal):
    if actor_goal is None:
        if isinstance(state, torch.Tensor):
            dummy = torch.zeros_like(state)
        else:
            dummy = np.zeros_like(state)
        return _concat_last_dim([state, dummy])
    return _concat_last_dim([state, actor_goal])


def sample_goals_from_batch(next_obs):
    state, _, critic_goal = split_obs_and_goal(next_obs)
    goals = critic_goal if critic_goal is not None else state
    if isinstance(goals, dict):
        goals = flatten_dict_obs(goals)
    if isinstance(goals, torch.Tensor):
        perm = torch.randperm(goals.shape[0], device=goals.device)
        return goals, goals[perm]
    perm = np.random.permutation(goals.shape[0])
    return goals, goals[perm]


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Actor(nn.Module):
    def __init__(self, envs, model):
        super().__init__()
        self.model = model

        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (envs.single_action_space.high - envs.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (envs.single_action_space.high + envs.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x, **kwargs):
        mean, log_std = self.model(x, **kwargs)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x, **kwargs):
        mean, log_std = self(x, **kwargs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class PhiEncoder(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class PsiEncoder(nn.Module):
    def __init__(self, goal_dim: int, hidden_dim: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        return self.net(goal)


@torch.no_grad()
def eval_agent(agent, test_env, num_evals, global_step, writer, device):
    obs, _ = test_env.reset()
    avg_ep_ret = 0
    avg_success = 0
    ep_ret = 0
    for _ in range(num_evals):
        while True:
            state, actor_goal, _ = split_obs_and_goal(obs)
            policy_input = build_policy_input(state, actor_goal)
            policy_input = torch.Tensor(policy_input).to(device).unsqueeze(0)
            action, _, _ = agent.get_action(policy_input)
            obs, reward, termination, truncation, info = test_env.step(
                action[0].cpu().numpy()
            )

            ep_ret += reward

            if termination or truncation:
                avg_success += info["success"]
                avg_ep_ret += ep_ret
                # resets
                obs, _ = test_env.reset()
                ep_ret = 0
                break
    avg_ep_ret /= num_evals
    avg_success /= num_evals
    print(f"\nTEST: ep_ret={avg_ep_ret}, success={avg_success}\n")
    writer.add_scalar("charts/test_episodic_return", avg_ep_ret, global_step)
    writer.add_scalar("charts/test_success", avg_success, global_step)
    if args.track:
        wandb.log({
            "charts/test_episodic_return": avg_ep_ret,
            "charts/test_success": avg_success,
        }, step=global_step)


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"task_{args.task_id}__{args.model_type}__{args.exp_name}__{args.seed}"
    print(f"\n*** Run name: {run_name} ***\n")

    writer = SummaryWriter(f"runs/{args.tag}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Initialize wandb tracking
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            tags=[args.model_type, args.tag, f"task_{args.task_id}"],
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"*** Device: {device}")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.task_id,
                args.capture_video,
                f"{args.tag}/{run_name}",
                args.video_every_n_episodes,
                args.video_dir,
            )
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    # select the model to use as the agent
    state_dim, goal_dim, critic_goal_dim = get_state_goal_dims(
        envs.single_observation_space
    )
    obs_dim = state_dim + goal_dim
    goal_embed_dim = critic_goal_dim if critic_goal_dim > 0 else state_dim
    act_dim = np.prod(envs.single_action_space.shape)
    print(state_dim)
    print(goal_dim)
    print(critic_goal_dim)
    print(act_dim)

    phi_encoder = PhiEncoder(
        state_dim=state_dim,
        action_dim=act_dim,
        hidden_dim=args.nce_hidden_dim,
        proj_dim=args.nce_proj_dim,
    ).to(device)
    psi_encoder = PsiEncoder(
        goal_dim=goal_embed_dim,
        hidden_dim=args.nce_hidden_dim,
        proj_dim=args.nce_proj_dim,
    ).to(device)

    print(f"*** Loading model `{args.model_type}` ***")
    if args.model_type in ["finetune", "componet"]:
        assert (
            len(args.prev_units) > 0
        ), f"Model type {args.model_type} requires at least one previous unit"

    if args.model_type == "simple":
        model = SimpleAgent(obs_dim=obs_dim, act_dim=act_dim).to(device)

    elif args.model_type == "finetune":
        model = SimpleAgent.load(
            args.prev_units[0], map_location=device, reset_heads=True
        ).to(device)

    elif args.model_type == "componet":
        model = CompoNetAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            prev_paths=args.prev_units,
            map_location=device,
        ).to(device)
    elif args.model_type == "packnet":
        packnet_retrain_start = args.total_timesteps - int(args.total_timesteps * 0.2)
        if len(args.prev_units) == 0:
            model = PackNetAgent(
                obs_dim=obs_dim,
                act_dim=act_dim,
                task_id=args.task_id,
                total_task_num=20,
                device=device,
            ).to(device)
        else:
            model = PackNetAgent.load(
                args.prev_units[0],
                task_id=args.task_id + 1,
                restart_heads=True,
                freeze_bias=True,
                map_location=device,
            ).to(device)
    elif args.model_type == "prognet":
        model = ProgressiveNetAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            prev_paths=args.prev_units,
            map_location=device,
        ).to(device)
    elif args.model_type == "cka-rl":
        base_dir = args.prev_units[0] if len(args.prev_units) > 0 else None
        latest_dir = args.prev_units[-1] if len(args.prev_units) > 0 else None
        model = CkaRlAgent(
            base_dir=base_dir,
            latest_dir=latest_dir,
            obs_dim=obs_dim,
            act_dim=act_dim,
            fuse_shared=False,
            fuse_heads=True,
            pool_size=args.pool_size,
            encoder_from_base=args.encoder_from_base,
        )
    elif args.model_type == 'masknet':
        if len(args.prev_units) == 0:
            model = MaskNetAgent(
                obs_dim=obs_dim,
                act_dim=act_dim,
                num_tasks=20,
            ).to(device)
        else:
            model = MaskNetAgent.load(
                args.prev_units[0],
                map_location=device,
            ).to(device)
        model.set_task(args.task_id, new_task=True)
    elif args.model_type == 'cbpnet':
        if len(args.prev_units) == 0:
            model = CbpAgent(
                obs_dim=obs_dim,
                act_dim=act_dim
                ).to(device)
        else:
            model = CbpAgent.load(
                args.prev_units[0],
                map_location=device,
            ).to(device)
    elif args.model_type == "crelus":
        if len(args.prev_units) == 0:
            model = CReLUsAgent(
                obs_dim=obs_dim,
                act_dim=act_dim
                ).to(device)
        else:
            model = CReLUsAgent.load(
                args.prev_units[0],
                map_location=device,
            ).to(device)
        
    actor = Actor(envs, model).to(device)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
    critic_optimizer = optim.Adam(
        list(phi_encoder.parameters()) + list(psi_encoder.parameters()),
        lr=args.q_lr,
    )
    if args.model_type == 'cbpnet':
        actor_optimizer = AdamGnT(actor.parameters(), lr=args.policy_lr, eps=1e-5)
        GnT = GnT(net=actor.model.fc.net, opt=actor_optimizer,replacement_rate=1e-3, decay_rate=0.99, device=device,
                    maturity_threshold=1000, util_type="contribution")

    if isinstance(envs.single_observation_space, gym.spaces.Box):
        envs.single_observation_space.dtype = np.float32
        rb = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            handle_timeout_termination=False,
        )
    else:
        rb = DictReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            handle_timeout_termination=False,
        )

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in tqdm(range(args.total_timesteps)):
        # ALGO LOGIC: put action logic here
        if global_step < args.random_actions_end:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            state, actor_goal, _ = split_obs_and_goal(obs)
            policy_input = build_policy_input(state, actor_goal)
            policy_input = torch.Tensor(policy_input).to(device)
            if args.model_type == "componet" and global_step % 1000 == 0:
                actions, _, _ = actor.get_action(
                    policy_input,
                    writer=writer,
                    global_step=global_step,
                )
            else:
                actions, _, _ = actor.get_action(policy_input)
            actions = actions.detach().cpu().numpy()
            if args.debug_print_interval > 0 and global_step % args.debug_print_interval == 0:
                print("=== Actor input (step) ===")
                print("state[0]:", np.asarray(state)[0] if np.asarray(state).ndim > 1 else np.asarray(state))
                if actor_goal is None:
                    print("goal: None (dummy used for policy input)")
                else:
                    print("goal[0]:", np.asarray(actor_goal)[0] if np.asarray(actor_goal).ndim > 1 else np.asarray(actor_goal))
                print("action[0]:", actions[0] if actions.ndim > 1 else actions)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for i, info in enumerate(infos["final_info"]):
                # print(
                #     f"global_step={global_step}, episodic_return={info['episode']['r']}, success={info['success']}"
                # )
                writer.add_scalar(
                    "charts/episodic_return", info["episode"]["r"], global_step
                )
                writer.add_scalar(
                    "charts/episodic_length", info["episode"]["l"], global_step
                )
                writer.add_scalar("charts/success", info["success"], global_step)
                if args.track:
                    wandb.log({
                        "charts/episodic_return": info["episode"]["r"],
                        "charts/episodic_length": info["episode"]["l"],
                        "charts/success": info["success"],
                    }, step=global_step)

                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            update_step = (
                global_step >= args.nce_start
                and global_step % args.nce_update_freq == 0
            )
            data = rb.sample(args.batch_size)
            nce_loss = None
            actor_loss = None
            pos_score = None
            neg_score = None

            if update_step:
                state, _, _ = split_obs_and_goal(data.observations)
                state = state.to(device) if isinstance(state, torch.Tensor) else torch.tensor(
                    state, device=device, dtype=torch.float32
                )
                state = state.float()
                goals_pos, _ = sample_goals_from_batch(data.next_observations)
                goals_pos = goals_pos.to(device) if isinstance(goals_pos, torch.Tensor) else torch.tensor(
                    goals_pos, device=device, dtype=torch.float32
                )
                goals_pos = goals_pos.float()
                actions = data.actions.float()

                phi = phi_encoder(state, actions)
                psi_pos = psi_encoder(goals_pos)
                logits = (phi @ psi_pos.T) / args.nce_temperature
                labels = torch.arange(logits.shape[0], device=logits.device)
                nce_loss = F.cross_entropy(logits, labels)
                pos_logits = logits.diag()
                pos_score = pos_logits.mean().item()
                if logits.shape[0] > 1:
                    neg_mask = ~torch.eye(
                        logits.shape[0], dtype=torch.bool, device=logits.device
                    )
                    neg_score = logits[neg_mask].mean().item()

                critic_optimizer.zero_grad()
                (args.nce_loss_weight * nce_loss).backward()
                critic_optimizer.step()

                policy_input = build_policy_input(state, goals_pos)
                pi, _, _ = actor.get_action(policy_input)
                phi_pi = phi_encoder(state, pi)
                psi_goal = psi_encoder(goals_pos).detach()
                actor_loss = -(phi_pi * psi_goal).sum(dim=-1).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                if args.model_type == "packnet":
                    if global_step >= packnet_retrain_start:
                        # can be called multiple times, only the first counts
                        actor.model.start_retraining()
                    actor.model.before_update()
                actor_optimizer.step()
                if args.debug_print_interval > 0 and global_step % args.debug_print_interval == 0:
                    print("=== Encoder batch (update) ===")
                    print("state[0]:", state[0].detach().cpu().numpy())
                    print("action[0]:", actions[0].detach().cpu().numpy())
                    print("goal_pos[0]:", goals_pos[0].detach().cpu().numpy())
                    row0 = logits[0].detach().cpu().numpy()
                    print("=== InfoNCE logits (row 0) ===")
                    print("pos_logit:", row0[0])
                    print("neg_logits:", row0[1: min(6, len(row0))])
                    print("nce_loss:", nce_loss.item())

            if global_step % 100 == 0:
                if nce_loss is not None:
                    writer.add_scalar("losses/nce_loss", nce_loss.item(), global_step)
                if actor_loss is not None:
                    writer.add_scalar(
                        "losses/actor_loss", actor_loss.item(), global_step
                    )
                if pos_score is not None:
                    writer.add_scalar("metrics/pos_score", pos_score, global_step)
                if neg_score is not None:
                    writer.add_scalar("metrics/neg_score", neg_score, global_step)
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.track:
                    log_dict = {
                        "charts/SPS": int(global_step / (time.time() - start_time)),
                    }
                    if nce_loss is not None:
                        log_dict["losses/nce_loss"] = nce_loss.item()
                    if actor_loss is not None:
                        log_dict["losses/actor_loss"] = actor_loss.item()
                    if pos_score is not None:
                        log_dict["metrics/pos_score"] = pos_score
                    if neg_score is not None:
                        log_dict["metrics/neg_score"] = neg_score
                    wandb.log(log_dict, step=global_step)
            if args.model_type == 'cbpnet':
                # print("cbpnet: selective initailization")
                GnT.gen_and_test(actor.model.fc.get_activations())
    [
        eval_agent(actor, envs.envs[i], args.num_evals, global_step, writer, device)
        for i in range(envs.num_envs)
    ]

    envs.close()
    writer.close()
    if args.track and args.capture_video:
        video_path = os.path.join(args.video_dir, args.tag, run_name)
        video_files = sorted(glob.glob(os.path.join(video_path, "*.mp4")))
        if video_files:
            for video_file in video_files[:3]:
                wandb.log({"videos": wandb.Video(video_file)}, step=global_step)
    if args.track:
        wandb.finish()

    if args.save_dir is not None:
        print(f"Saving trained agent in `{args.save_dir}` with name `{run_name}`")
        actor.model.save(dirname=f"{args.save_dir}/{run_name}")
