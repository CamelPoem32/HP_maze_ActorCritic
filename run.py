import argparse
import torch
import time
from env import HarryPotterEnv
from models import ActorCriticNet
from a2c_separate import ActorNet

# Assuming visualize_episode is imported from your visualize.py module
from visualize import visualize_episode 

def main():
    parser = argparse.ArgumentParser(description="Visualize Trained Harry Potter RL Agents")
    parser.add_argument('--method', type=str, required=True, 
                        choices=['a2c_shared', 'ppo', 'a2c_separate'], 
                        help='Which architecture/algorithm was used to train the checkpoint.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the .pt checkpoint file')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = args.device
    print(f"Using device: {device}")

    env = HarryPotterEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Wrapper class for the separate actor to match the visualizer's expected API
    class SeparateActorWrapper:
        def __init__(self, actor):
            self.actor = actor
        def get_action(self, obs):
            mean, _ = self.actor(obs)
            return mean, None, None 

    if args.method in ['a2c_shared', 'ppo']:
        model = ActorCriticNet(obs_dim, act_dim).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        agent = model
    elif args.method == 'a2c_separate':
        actor = ActorNet(obs_dim, act_dim).to(device)
        actor.load_state_dict(checkpoint['actor_state_dict'])
        actor.eval()
        agent = SeparateActorWrapper(actor)

    print(f"Starting visualization for {args.method}...")
    
    # We need a slight modification to the visualization loop to handle the device
    obs, _ = env.reset()
    done = False
    
    # Optional: If your visualize_episode function is exactly the one from Cell 5 earlier, 
    # it assumes CPU. We do the forward pass here to keep it clean.
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.ion()
    
    while not done:
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        # Draw Walls
        for w in env.walls:
            rect = patches.Rectangle((w[0], w[2]), w[1]-w[0], w[3]-w[2], fill=True, color='gray')
            ax.add_patch(rect)
            
        # Draw Goal & Entities
        goal = patches.Circle((env.goal_pos[0], env.goal_pos[1]), env.goal_radius, color='gold', label='Exit')
        harry = patches.Circle((env.harry_pos[0], env.harry_pos[1]), 0.2, color='blue', label='Harry')
        filch = patches.Circle((env.filch_pos[0], env.filch_pos[1]), 0.2, color='brown', label='Filch')
        cat = patches.Circle((env.cat_pos[0], env.cat_pos[1]), 0.2, color='red', label='Mrs. Norris')
        cat_smell = patches.Circle((env.cat_pos[0], env.cat_pos[1]), env.smell_radius, color='red', alpha=0.1)
        # Sight radius representation
        harry_sight = patches.Circle((env.harry_pos[0], env.harry_pos[1]), env.sight_radius, color='blue', alpha=0.1)
        
        ax.add_patch(goal)
        ax.add_patch(harry)
        ax.add_patch(filch)
        ax.add_patch(cat)
        ax.add_patch(cat_smell)
        ax.add_patch(harry_sight)
        
        ax.legend(loc='upper right')
        plt.pause(0.05)
        
        # Prepare observation for the model
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action, _, _ = agent.get_action(obs_tensor)
            
        # Move action back to CPU to interact with Env
        obs, reward, done, _, info = env.step(action.detach().cpu().squeeze().numpy())
        
    plt.ioff()
    print(f"Episode finished. Result: {info.get('result', 'N/A')}")
    plt.show()

if __name__ == '__main__':
    # Run with
    # python run.py --method ppo --checkpoint ppo/ppo_10k.pt
    # python run.py --method a2c_shared --checkpoint a2c/a2c_10k.pt
    # python run.py --method a2c_separate --checkpoint a2c_separate/a2c_separate_10k.pt
    main()