# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
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
from models import shared, SimpleAgent, CompoNetAgent, PackNetAgent, ProgressiveNetAgent, CkaRlAgent, MaskNetAgent, CbpAgent, CReLUsAgent
from tasks import get_task
from utils.AdamGnT import AdamGnT
from stable_baselines3.common.buffers import ReplayBuffer
from models.cbp_modules import GnT


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
    wandb_project_name: str = "cw-sac"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    task_id: int = 0
    """ID number of the task"""
    eval_every: int = 10_000
    """Evaluate the agent in determinstic mode every X timesteps"""
    num_evals: int = 10
    """Number of times to evaluate the agent"""
    total_timesteps: int = int(1e5)
    """total timesteps of the experiments (100k for REDQ due to high sample efficiency)"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5_000
    """timestep to start learning"""
    random_actions_end: int = 10_000
    """timesteps to take actions randomly"""
    policy_lr: float = 1e-3
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    tag: str = "Debug"
    """experiment tag"""
    pool_size: int = 9
    """pool size"""
    encoder_from_base: bool = False
    """load encoder from base_dir"""
    num_q_nets: int = 10
    """Number of Q-networks in the ensemble (REDQ)"""
    q_target_subset_size: int = 2
    """Number of Q-networks to randomly select for target computation (REDQ)"""
    utd_ratio: int = 20
    """Update-to-data ratio: number of gradient updates per environment step (REDQ)"""

def make_env(task_id):
    def thunk():
        env = get_task(task_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.fc = shared(
            np.array(envs.observation_space.shape).prod()
            + np.prod(envs.action_space.shape)
        )
        self.fc_out = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.fc(x)
        x = self.fc_out(x)
        return x


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


@torch.no_grad()
def eval_agent(agent, test_env, num_evals, global_step, writer, device):
    obs, _ = test_env.reset()
    avg_ep_ret = 0
    avg_success = 0
    ep_ret = 0
    for _ in range(num_evals):
        while True:
            obs = torch.Tensor(obs).to(device).unsqueeze(0)
            action, _ = agent(obs)
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
    envs = gym.vector.SyncVectorEnv([make_env(args.task_id)])
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    # select the model to use as the agent
    obs_dim = np.array(envs.single_observation_space.shape).prod()
    act_dim = np.prod(envs.single_action_space.shape)
    print(obs_dim)
    print(act_dim)
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
    # REDQ: Create ensemble of N Q-networks
    q_nets = [SoftQNetwork(envs).to(device) for _ in range(args.num_q_nets)]
    q_targets = [SoftQNetwork(envs).to(device) for _ in range(args.num_q_nets)]
    # Initialize target networks with main network weights
    for q_target in q_targets:
        q_target.load_state_dict(q_nets[0].state_dict())
    # REDQ: Optimizer for all Q-networks
    q_optimizer = optim.Adam(
        [p for q_net in q_nets for p in q_net.parameters()], lr=args.q_lr
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
    if args.model_type == 'cbpnet':
        actor_optimizer = AdamGnT(actor.parameters(), lr=args.policy_lr, eps=1e-5)
        GnT = GnT(net=actor.model.fc.net, opt=actor_optimizer,replacement_rate=1e-3, decay_rate=0.99, device=device,
                    maturity_threshold=1000, util_type="contribution")
        
    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(
            torch.Tensor(envs.action_space.shape).to(device)
        ).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
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
            if args.model_type == "componet" and global_step % 1000 == 0:
                actions, _, _ = actor.get_action(
                    torch.Tensor(obs).to(device),
                    writer=writer,
                    global_step=global_step,
                )
            else:
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

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
        # REDQ: UTD ratio - perform multiple updates per environment step
        if global_step > args.learning_starts:
            for utd_iter in range(args.utd_ratio):
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = actor.get_action(
                        data.next_observations
                    )
                    # REDQ: Randomly select subset of Q-networks for target computation
                    selected_indices = np.random.choice(
                        args.num_q_nets, args.q_target_subset_size, replace=False
                    )
                    q_next_targets = [
                        q_targets[i](data.next_observations, next_state_actions)
                        for i in selected_indices
                    ]
                    # Take minimum of selected Q-networks
                    min_qf_next_target = (
                        torch.min(torch.stack(q_next_targets), dim=0)[0]
                        - alpha * next_state_log_pi
                    )
                    next_q_value = data.rewards.flatten() + (
                        1 - data.dones.flatten()
                    ) * args.gamma * (min_qf_next_target).view(-1)

                # REDQ: Compute loss for all Q-networks in ensemble
                q_losses = []
                q_a_values_list = []
                for q_net in q_nets:
                    q_a_values = q_net(data.observations, data.actions).view(-1)
                    q_loss = F.mse_loss(q_a_values, next_q_value)
                    q_losses.append(q_loss)
                    q_a_values_list.append(q_a_values)
                qf_loss = sum(q_losses)

                # optimize the model
                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                    for _ in range(
                        args.policy_frequency
                    ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                        pi, log_pi, _ = actor.get_action(data.observations)
                        # REDQ: Use all Q-networks for actor loss (take average, not minimum)
                        q_pi_values = [q_net(data.observations, pi) for q_net in q_nets]
                        mean_qf_pi = torch.stack(q_pi_values).mean(dim=0)
                        actor_loss = ((alpha * log_pi) - mean_qf_pi).mean()

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        if args.model_type == "packnet":
                            if global_step >= packnet_retrain_start:
                                # can be called multiple times, only the first counts
                                actor.model.start_retraining()
                            actor.model.before_update()
                        actor_optimizer.step()

                        if args.autotune:
                            with torch.no_grad():
                                _, log_pi, _ = actor.get_action(data.observations)
                            alpha_loss = (
                                -log_alpha.exp() * (log_pi + target_entropy)
                            ).mean()

                            a_optimizer.zero_grad()
                            alpha_loss.backward()
                            a_optimizer.step()
                            alpha = log_alpha.exp().item()

                # REDQ: update all target networks
                if global_step % args.target_network_frequency == 0:
                    for q_net, q_target in zip(q_nets, q_targets):
                        for param, target_param in zip(
                            q_net.parameters(), q_target.parameters()
                        ):
                            target_param.data.copy_(
                                args.tau * param.data
                                + (1 - args.tau) * target_param.data
                            )

                # REDQ: Log aggregate statistics for ensemble of Q-networks (only on first UTD iteration when logging)
                if global_step % 100 == 0 and utd_iter == 0:
                    q_values_mean = torch.stack(q_a_values_list).mean().item()
                    q_values_std = torch.stack(q_a_values_list).std().item()
                    q_loss_mean = sum(q_losses).item() / len(q_losses)
                    writer.add_scalar("losses/qf_mean_values", q_values_mean, global_step)
                    writer.add_scalar("losses/qf_std_values", q_values_std, global_step)
                    writer.add_scalar("losses/qf_loss", q_loss_mean, global_step)
                    writer.add_scalar("losses/alpha", alpha, global_step)
                    # print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar(
                        "charts/SPS",
                        int(global_step / (time.time() - start_time)),
                        global_step,
                    )
                    if args.track:
                        log_dict = {
                            "losses/qf_mean_values": q_values_mean,
                            "losses/qf_std_values": q_values_std,
                            "losses/qf_loss": q_loss_mean,
                            "losses/alpha": alpha,
                            "charts/SPS": int(global_step / (time.time() - start_time)),
                        }
                        wandb.log(log_dict, step=global_step)
                
                # Log actor_loss and alpha_loss only when policy is updated
                if global_step % args.policy_frequency == 0 and global_step % 100 == 0 and utd_iter == 0:
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    if args.autotune:
                        writer.add_scalar(
                            "losses/alpha_loss", alpha_loss.item(), global_step
                        )
                    if args.track:
                        log_dict = {
                            "losses/actor_loss": actor_loss.item(),
                        }
                        if args.autotune:
                            log_dict["losses/alpha_loss"] = alpha_loss.item()
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
    if args.track:
        wandb.finish()

    if args.save_dir is not None:
        print(f"Saving trained agent in `{args.save_dir}` with name `{run_name}`")
        actor.model.save(dirname=f"{args.save_dir}/{run_name}")
