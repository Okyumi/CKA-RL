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
import math
import tyro
import pathlib
from torch.utils.tensorboard import SummaryWriter
from typing import Literal, Optional, Tuple
import wandb
from models import SimpleAgent, CompoNetAgent, PackNetAgent, ProgressiveNetAgent, CkaRlAgent, MaskNetAgent, CbpAgent, CReLUsAgent
from tasks import get_task
from utils.AdamGnT import AdamGnT
from models.cbp_modules import GnT
from wrapper.goal_wrapper import GoalObsWrapper
from buffer import TrajectoryBuffer


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
    num_envs: int = 1
    """number of parallel environments"""
    total_timesteps: int = 100_000_000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    learning_starts: int = 512_000
    """timestep to start learning"""
    random_actions_end: int = 512_000
    """timesteps to take actions randomly"""
    policy_lr: float = 1e-3
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the contrastive encoders"""
    alpha: float = 0.2
    """entropy regularization coefficient"""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
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
    nce_hidden_dim: int = 1024
    """hidden dimension for contrastive encoder"""
    critic_network_depth: int = 16
    """depth (multiple of 4) for critic encoders"""
    nce_update_freq: int = 40
    """update contrastive loss every N steps"""
    nce_start: int = 5_000
    """global step to start contrastive updates"""
    logsumexp_penalty_coeff: float = 1e-3
    """logsumexp regularization coefficient for critic loss"""
    actor_network_width: int = 1024
    """hidden width for actor shared trunk"""
    actor_network_depth: int = 16
    """depth (multiple of 4) for actor shared trunk"""
    num_episodes_per_env: int = 1
    """number of trajectory batches sampled per training step"""
    num_sgd_batches_per_training_step: int = 1
    """number of SGD batches per training step when subsampling"""
    use_all_batches: int = 1
    """if 0, subsample SGD batches per training step"""
    debug_print_interval: int = 100
    """print state/action/goal samples every N steps (0 disables)"""

def make_env(task_id, capture_video, run_name, video_every_n_episodes, video_dir, augment_goal=True):
    def thunk():
        env = get_task(task_id)
        if capture_video:
            env.render_mode = "rgb_array"
        env = GoalObsWrapper(env, goal_dim=3, augment_goal=augment_goal)
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
    state_space = obs_space["observation"]
    goal_space = obs_space["desired_goal"]
    critic_goal_space = obs_space["critic_goal"]
    state_dim = int(np.prod(state_space.shape))
    goal_dim = int(np.prod(goal_space.shape))
    critic_goal_dim = int(np.prod(critic_goal_space.shape))
    return state_dim, goal_dim, critic_goal_dim


def _concat_last_dim(parts):
    if isinstance(parts[0], torch.Tensor):
        return torch.cat(parts, dim=-1)
    return np.concatenate(parts, axis=-1)


def split_obs_and_goal(obs):
    """Extract state, actor goal, and critic goal from wrapped env outputs."""
    return obs["observation"], obs["desired_goal"], obs["critic_goal"]


def build_policy_input(state, actor_goal):
    return _concat_last_dim([state, actor_goal])




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


def lecun_uniform_init(tensor):
    """LeCun uniform initialization: variance_scaling(1/3, "fan_in", "uniform")"""
    fan_in = tensor.size(1)
    bound = math.sqrt(1.0 / (3.0 * fan_in))
    with torch.no_grad():
        tensor.uniform_(-bound, bound)


class ResidualBlock(nn.Module):
    """Residual block with 4 linear layers, following the paper's structure."""
    def __init__(self, width: int, use_layer_norm: bool = True, use_relu: bool = False):
        super().__init__()
        self.width = width
        self.use_layer_norm = use_layer_norm
        self.activation = nn.ReLU() if use_relu else nn.SiLU()
        
        # 4 linear layers
        self.layers = nn.ModuleList([
            nn.Linear(width, width) for _ in range(4)
        ])
        
        # Normalization layers
        if use_layer_norm:
            self.norms = nn.ModuleList([
                nn.LayerNorm(width) for _ in range(4)
            ])
        else:
            self.norms = nn.ModuleList([nn.Identity() for _ in range(4)])
        
        # Initialize weights
        for layer in self.layers:
            lecun_uniform_init(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x)
            x = norm(x)
            x = self.activation(x)
        x = x + identity
        return x


class PhiEncoder(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        proj_dim: int,
        depth: int = 4,
        use_layer_norm: bool = True,
        use_relu: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm
        self.activation = nn.ReLU() if use_relu else nn.SiLU()
        
        # Initial layer
        self.initial_layer = nn.Linear(state_dim + action_dim, hidden_dim)
        if use_layer_norm:
            self.initial_norm = nn.LayerNorm(hidden_dim)
        else:
            self.initial_norm = nn.Identity()
        
        # Residual blocks (4 layers each)
        num_blocks = max(1, depth // 4)
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, use_layer_norm, use_relu) for _ in range(num_blocks)]
        )
        
        # Final layer
        self.final_layer = nn.Linear(hidden_dim, proj_dim)
        
        # Initialize weights
        lecun_uniform_init(self.initial_layer.weight)
        nn.init.zeros_(self.initial_layer.bias)
        lecun_uniform_init(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        # Initial layer
        x = self.initial_layer(x)
        x = self.initial_norm(x)
        x = self.activation(x)
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        # Final layer
        x = self.final_layer(x)
        return x


class PsiEncoder(nn.Module):
    def __init__(
        self,
        goal_dim: int,
        hidden_dim: int,
        proj_dim: int,
        depth: int = 4,
        use_layer_norm: bool = True,
        use_relu: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm
        self.activation = nn.ReLU() if use_relu else nn.SiLU()
        
        # Initial layer
        self.initial_layer = nn.Linear(goal_dim, hidden_dim)
        if use_layer_norm:
            self.initial_norm = nn.LayerNorm(hidden_dim)
        else:
            self.initial_norm = nn.Identity()
        
        # Residual blocks (4 layers each)
        num_blocks = max(1, depth // 4)
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, use_layer_norm, use_relu) for _ in range(num_blocks)]
        )
        
        # Final layer
        self.final_layer = nn.Linear(hidden_dim, proj_dim)
        
        # Initialize weights
        lecun_uniform_init(self.initial_layer.weight)
        nn.init.zeros_(self.initial_layer.bias)
        lecun_uniform_init(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        x = goal
        # Initial layer
        x = self.initial_layer(x)
        x = self.initial_norm(x)
        x = self.activation(x)
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        # Final layer
        x = self.final_layer(x)
        return x


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
    envs = gym.vector.AsyncVectorEnv(
        [
            make_env(
                args.task_id,
                args.capture_video,
                f"{args.tag}/{run_name}",
                args.video_every_n_episodes,
                args.video_dir,
                augment_goal=True,
            )
            for _ in range(args.num_envs)
        ]
    )

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

    # Load or create critic encoders
    phi_encoder = None
    psi_encoder = None
    
    # Try to load encoders from previous task if available
    if len(args.prev_units) > 0:
        latest_dir = args.prev_units[-1]
        phi_path = f"{latest_dir}/phi_encoder.pt"
        psi_path = f"{latest_dir}/psi_encoder.pt"
        
        if os.path.exists(phi_path) and os.path.exists(psi_path):
            print(f"*** Loading critic encoders from {latest_dir} ***")
            try:
                phi_encoder = torch.load(phi_path, map_location=device)
                psi_encoder = torch.load(psi_path, map_location=device)
                print("Successfully loaded critic encoders from previous task")
            except Exception as e:
                print(f"Warning: Failed to load critic encoders: {e}")
                print("Initializing new critic encoders")
                phi_encoder = None
                psi_encoder = None
    
    # Create new encoders if not loaded
    if phi_encoder is None:
        print("*** Initializing new critic encoders ***")
        phi_encoder = PhiEncoder(
            state_dim=state_dim,
            action_dim=act_dim,
            hidden_dim=args.nce_hidden_dim,
            proj_dim=args.nce_proj_dim,
            depth=args.critic_network_depth,
        ).to(device)
        psi_encoder = PsiEncoder(
            goal_dim=goal_embed_dim,
            hidden_dim=args.nce_hidden_dim,
            proj_dim=args.nce_proj_dim,
            depth=args.critic_network_depth,
        ).to(device)

    print(f"*** Loading model `{args.model_type}` ***")
    if args.model_type in ["finetune", "componet"]:
        assert (
            len(args.prev_units) > 0
        ), f"Model type {args.model_type} requires at least one previous unit"

    if args.model_type == "simple":
        model = SimpleAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            network_width=args.actor_network_width,
            network_depth=args.actor_network_depth,
        ).to(device)

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
            network_width=args.actor_network_width,
            network_depth=args.actor_network_depth,
        )
    elif args.model_type == 'masknet':
        if len(args.prev_units) == 0:
            model = MaskNetAgent(
                obs_dim=obs_dim,
                act_dim=act_dim,
                num_tasks=20,
                network_width=args.actor_network_width,
                network_depth=args.actor_network_depth,
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

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(
            torch.Tensor(envs.single_action_space.shape).to(device)
        ).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha
    alpha_state = {"value": alpha}

    # Initialize trajectory-aware replay buffer for contrastive RL
    # This buffer automatically samples future goals from the same trajectory
    rb = TrajectoryBuffer(
        buffer_size=args.buffer_size,
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
        device=device,
        episode_length=500,  # Typical MetaWorld episode length
        gamma=0.99,  # Discount factor for geometric distribution
        goal_start_idx=4,  # Goal indices [4, 5, 6] for MetaWorld
        goal_end_idx=7,
    )

    start_time = time.time()

    def update_actor(state, goals_pos, global_step):
        def actor_loss(policy_input):
            pi, log_pi, _ = actor.get_action(policy_input)
            phi_pi = phi_encoder(state, pi)
            psi_goal = psi_encoder(goals_pos).detach()
            qf_pi = -torch.sqrt(torch.sum((phi_pi - psi_goal) ** 2, dim=-1))
            entropy_loss = (alpha_state["value"] * log_pi).mean()
            return entropy_loss - qf_pi.mean()

        def alpha_loss(policy_input):
            if not args.autotune:
                return None
            with torch.no_grad():
                _, log_pi, _ = actor.get_action(policy_input)
            return (-log_alpha.exp() * (log_pi + target_entropy)).mean()

        policy_input = build_policy_input(state, goals_pos)
        loss = actor_loss(policy_input)

        actor_optimizer.zero_grad()
        loss.backward()
        if args.model_type == "packnet":
            if global_step >= packnet_retrain_start:
                # can be called multiple times, only the first counts
                actor.model.start_retraining()
            actor.model.before_update()
        actor_optimizer.step()

        alpha_loss_value = alpha_loss(policy_input)
        if alpha_loss_value is not None:
            a_optimizer.zero_grad()
            alpha_loss_value.backward()
            a_optimizer.step()
            alpha_state["value"] = log_alpha.exp().item()

        return loss, alpha_loss_value

    def update_critic(state, goals_pos, actions, trajectory_ids, global_step):
        phi = phi_encoder(state, actions)
        psi_pos = psi_encoder(goals_pos)

        # DEBUG: Check trajectory distribution and encoding diversity
        if args.debug_print_interval > 0 and global_step % args.debug_print_interval == 0:
            unique_trajectories = torch.unique(trajectory_ids)
            print(f"=== DEBUG: Batch Analysis ===")
            print(f"Batch size: {len(trajectory_ids)}")
            print(f"Unique trajectories: {len(unique_trajectories)}")
            print(f"Trajectory IDs: {trajectory_ids[:10].cpu().numpy()}")
            print(f"Phi encoding stats - mean: {phi.mean().item():.6f}, std: {phi.std().item():.6f}")
            print(f"Psi encoding stats - mean: {psi_pos.mean().item():.6f}, std: {psi_pos.std().item():.6f}")
            # Check if encodings are all the same
            phi_diff = (phi[0:1] - phi[1:2]).abs().mean().item()
            psi_diff = (psi_pos[0:1] - psi_pos[1:2]).abs().mean().item()
            print(f"Phi encoding difference (first 2): {phi_diff:.6f}")
            print(f"Psi encoding difference (first 2): {psi_diff:.6f}")

        # Compute similarity as negative L2 distance (as per diagram)
        # sim_ij = -||φ(s_i, a_i) - ψ(g_j)||_2
        phi_expanded = phi.unsqueeze(1)  # [batch_size, 1, proj_dim]
        psi_expanded = psi_pos.unsqueeze(0)  # [1, batch_size, proj_dim]
        distances = torch.norm(phi_expanded - psi_expanded, dim=2)  # [batch_size, batch_size]
        logits = -distances / args.nce_temperature

        logsumexp = torch.logsumexp(logits, dim=1)
        nce_loss = -(logits.diag() - logsumexp).mean()
        nce_loss = nce_loss + args.logsumexp_penalty_coeff * (logsumexp.pow(2).mean())
        pos_score = logits.diag().mean().item()
        neg_score = None
        if logits.shape[0] > 1:
            neg_mask = ~torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
            neg_score = logits[neg_mask].mean().item()

        critic_optimizer.zero_grad()
        (args.nce_loss_weight * nce_loss).backward()
        critic_optimizer.step()
        return nce_loss, pos_score, neg_score, logits

    def sgd_step(state, goals_pos, actions, trajectory_ids, global_step):
        actor_loss, alpha_loss = update_actor(state, goals_pos, global_step)
        nce_loss, pos_score, neg_score, logits = update_critic(
            state, goals_pos, actions, trajectory_ids, global_step
        )
        return nce_loss, pos_score, neg_score, logits, actor_loss, alpha_loss

    def _concat_batches(batches):
        observations = torch.cat([b.observations.observation for b in batches], dim=0)
        critic_goals = torch.cat([b.observations.critic_goal for b in batches], dim=0)
        actions = torch.cat([b.actions for b in batches], dim=0)
        trajectory_ids = torch.cat([b.trajectory_ids for b in batches], dim=0)
        return observations, critic_goals, actions, trajectory_ids

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
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos, truncations)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            update_step = (
                global_step >= args.nce_start
                and global_step % args.nce_update_freq == 0
            )
            nce_loss = None
            actor_loss = None
            alpha_loss = None
            pos_score = None
            neg_score = None

            if update_step:
                batches = [rb.sample(args.batch_size) for _ in range(args.num_episodes_per_env)]
                state, goals_pos, actions, trajectory_ids = _concat_batches(batches)
                total_samples = state.shape[0]
                perm = torch.randperm(total_samples, device=state.device)
                state = state[perm]
                goals_pos = goals_pos[perm]
                actions = actions[perm]
                trajectory_ids = trajectory_ids[perm]

                num_full_batches = total_samples // args.batch_size
                if num_full_batches == 0:
                    continue
                limit = num_full_batches * args.batch_size
                state = state[:limit]
                goals_pos = goals_pos[:limit]
                actions = actions[:limit]
                trajectory_ids = trajectory_ids[:limit]

                state = state.view(num_full_batches, args.batch_size, -1)
                goals_pos = goals_pos.view(num_full_batches, args.batch_size, -1)
                actions = actions.view(num_full_batches, args.batch_size, -1)
                trajectory_ids = trajectory_ids.view(num_full_batches, args.batch_size)

                batch_indices = torch.arange(num_full_batches, device=state.device)
                if args.use_all_batches == 0:
                    num_select = min(args.num_sgd_batches_per_training_step, num_full_batches)
                    batch_indices = batch_indices[torch.randperm(num_full_batches, device=state.device)[:num_select]]

                for batch_idx in batch_indices:
                    nce_loss, pos_score, neg_score, logits, actor_loss, alpha_loss = sgd_step(
                        state[batch_idx],
                        goals_pos[batch_idx],
                        actions[batch_idx],
                        trajectory_ids[batch_idx],
                        global_step,
                    )
                if args.debug_print_interval > 0 and global_step % args.debug_print_interval == 0:
                    print("=== Encoder batch (update) ===")
                    print("state[0]:", state[0][0].detach().cpu().numpy())
                    print("action[0]:", actions[0][0].detach().cpu().numpy())
                    print("goal_pos[0]:", goals_pos[0][0].detach().cpu().numpy())
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
                writer.add_scalar("losses/alpha", alpha_state["value"], global_step)
                if args.autotune and alpha_loss is not None:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
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
                    log_dict["losses/alpha"] = alpha_state["value"]
                    if args.autotune and alpha_loss is not None:
                        log_dict["losses/alpha_loss"] = alpha_loss.item()
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
        save_path = f"{args.save_dir}/{run_name}"
        print(f"Saving trained agent in `{save_path}`")
        actor.model.save(dirname=save_path)
        # Save critic encoders for next task
        print(f"Saving critic encoders in `{save_path}`")
        torch.save(phi_encoder, f"{save_path}/phi_encoder.pt")
        torch.save(psi_encoder, f"{save_path}/psi_encoder.pt")
