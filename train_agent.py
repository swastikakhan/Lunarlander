import gymnasium as gym
import numpy as np
import argparse
import os
import multiprocessing

def policy_action(params, observation):
    """Maps observations to actions using a linear policy."""
    W = params[:8 * 4].reshape(8, 4)
    b = params[8 * 4:].reshape(4)
    logits = np.dot(observation, W) + b
    return np.argmax(logits)

def evaluate_single_episode(params, seed=None):
    """Runs a single episode on a given seed and returns the total reward."""
    env = gym.make('LunarLander-v3')
    observation, info = env.reset(seed=int(seed))
    episode_reward = 0.0
    done = False
    while not done:
        action = policy_action(params, observation)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated
    env.close()
    return episode_reward

def evaluate_policy(params, episodes=10):
    """Runs policy evaluation with different terrain seeds."""
    seeds = np.random.randint(1000, size=episodes)  # Generate different seeds
    rewards = [evaluate_single_episode(params, seed=seed) for seed in seeds]
    return np.mean(rewards)

def particle_swarm_optimization(filename, population_size=300, num_generations=100000, w=0.7, c1=1.5, c2=3,
                                lower_bound=-10, upper_bound=10):
    """Optimizes policy parameters using Particle Swarm Optimization (PSO) with multiprocessing."""
    gene_size = 8 * 4 + 4  # 8 inputs x 4 outputs + 4 biases = 36 parameters
    swarm = []

    # Initialize particles
    for _ in range(population_size):
        particle = {
            "position": np.random.uniform(lower_bound, upper_bound, gene_size),
            "velocity": np.random.uniform(-1, 1, gene_size),
            "pbest": None,
            "pbest_value": -np.inf
        }
        particle["pbest"] = np.copy(particle["position"])
        swarm.append(particle)

    gbest_position = None
    gbest_value = -np.inf
    avg = -np.inf
    for generation in range(num_generations):
        if gbest_value >= 290:
            w = 0.3
            c1 = 1
        # Evaluate all particles in parallel (top-level multiprocessing)
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            fitness_values = pool.map(evaluate_policy, [p["position"] for p in swarm])
        sum = 0
        for i, particle in enumerate(swarm):
            fitness = fitness_values[i]
            sum += fitness

            # Update personal best
            if fitness > particle["pbest_value"]:
                particle["pbest"] = np.copy(particle["position"])
                particle["pbest_value"] = fitness

            # Update global best and save
            if fitness > gbest_value:
                gbest_position = np.copy(particle["position"])
                gbest_value = fitness
                np.save(filename, gbest_position)
                print(f"New best policy saved at generation {generation + 1} with reward {gbest_value:.2f}")
            
        if generation % 100 == 0:
            np.save(filename, gbest_position)
            print(f"Best policy saved at generation {generation + 1} with reward {gbest_value:.2f}")
        print(f"Avg score of all particles = {sum/population_size}")
        if avg < sum/population_size:
            avg = sum/population_size
            np.save(filename, gbest_position)
            print(f"Best policy saved at generation {generation + 1} with reward {gbest_value:.2f}")
        sum = 0

        # Update particles
        for particle in swarm:
            r1, r2 = np.random.rand(gene_size), np.random.rand(gene_size)

            inertia = w * particle["velocity"]
            cognitive = c1 * r1 * (particle["pbest"] - particle["position"])
            social = c2 * r2 * (gbest_position - particle["position"])

            particle["velocity"] = inertia + cognitive + social
            particle["position"] += particle["velocity"]

            # Ensure position is within bounds
            particle["position"] = np.clip(particle["position"], lower_bound, upper_bound)

        print(f"Generation {generation + 1}: Best Reward = {gbest_value:.2f}")

    return gbest_position

def train_and_save(filename, **kwargs):
    """Trains the policy using PSO and saves the best policy parameters."""
    best_params = particle_swarm_optimization(filename, **kwargs)
    print(f"Best policy saved to {filename}")
    return best_params

def load_policy(filename):
    """Loads policy parameters from a file."""
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        return None
    best_params = np.load(filename)
    print(f"Loaded best policy from {filename}")
    return best_params

def play_policy(best_params, episodes=5):
    """Plays the Lunar Lander game using the loaded policy parameters."""
    seeds = np.random.randint(1000, size=episodes)
    rewards = [evaluate_single_episode(best_params, seed=seed) for seed in seeds]
    avg_reward = np.mean(rewards)
    print(f"Average reward of the best policy over {episodes} episodes: {avg_reward:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or play best policy for Lunar Lander using PSO.")
    parser.add_argument("--train", action="store_true", help="Train the policy using PSO and save it.")
    parser.add_argument("--play", action="store_true", help="Load the best policy and play.")
    parser.add_argument("--filename", type=str, default="best_policy.npy", help="Filename to save/load the best policy.")
    parser.add_argument("--pop_size", type=int, default=300, help="Population size for PSO.")
    parser.add_argument("--generations", type=int, default=100000, help="Number of generations for PSO.")
    parser.add_argument("--w", type=float, default=0.7, help="Inertia weight for PSO.")
    parser.add_argument("--c1", type=float, default=1.5, help="Cognitive coefficient for PSO.")
    parser.add_argument("--c2", type=float, default=3,  help="Social coefficient for PSO.")
    parser.add_argument("--lower_bound", type=float, default=-10, help="Lower bound for PSO parameters.")
    parser.add_argument("--upper_bound", type=float, default=10, help="Upper bound for PSO parameters.")

    args = parser.parse_args()

    if args.train:
        best_params = train_and_save(
            filename=args.filename,
            population_size=args.pop_size,
            num_generations=args.generations,
            w=args.w,
            c1=args.c1,
            c2=args.c2,
            lower_bound=args.lower_bound,
            upper_bound=args.upper_bound
        )
    elif args.play:
        best_params = load_policy(args.filename)
        if best_params is not None:
            play_policy(best_params, episodes=5)
        else:
            print("Please specify --train to train and save a policy, or --play to load and play the best policy.")
    else:
        print("Please specify --train to train and save a policy, or --play to load and play the best policy.")
