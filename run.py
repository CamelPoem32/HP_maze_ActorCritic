import argparse
import torch
import time
from env_cnn import HarryPotterEnv
from models_cnn import ActorCriticNet, ActorNet, SharedACNet
# from a2c_separate_cnn import ActorNet
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def main():
    parser = argparse.ArgumentParser(description="Visualize Trained Harry Potter RL Agents")
    parser.add_argument('--method', type=str, required=True, 
                        choices=['a2c_shared', 'ppo', 'a2c_separate'], 
                        help='Which algorithm was used.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pt file')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    env = HarryPotterEnv()
    obs, _ = env.reset()
    
    # CNN shapes: (Channels, H, W)
    obs_shape = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # 1. Setup Models
    if args.method in ['a2c_shared', 'ppo']:
        agent = ActorCriticNet(obs_shape, act_dim).to(device)
        agent.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Separate A2C with Shared CNN (using SharedACNet class)
        # Note: We use SharedACNet because the Actor and Critic now live in one "body"
        shared_model = SharedACNet(obs_shape, act_dim).to(device).float()
        
        # Check your training script: if you saved it as 'model_state_dict', use that.
        # If you saved it as 'actor_state_dict'/ 'critic_state_dict', you'll need to 
        # load the specific one used during training.
        if 'model_state_dict' in checkpoint:
            shared_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            shared_model.load_state_dict(checkpoint['actor_state_dict'])

        class SharedSeparateWrapper:
            def __init__(self, model):
                self.model = model
                
            def eval(self):
                self.model.eval()
                
            def get_action(self, obs_tensor):
                obs_tensor = obs_tensor.to(dtype=torch.float32)
                with torch.no_grad():
                    # Unpack 3 values: mean, std, and the value (which we ignore here)
                    mean, std, value = self.model(obs_tensor)
                return mean, None, None
                
        agent = SharedSeparateWrapper(shared_model)
    
    agent.eval()

    # 2. Optimized Matplotlib Setup
    plt.close('all') # Close any ghost windows
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Ensure window stays on top and is interactive
    plt.ion()
    fig.show()

    done = False
    step_count = 0

    try:
        while not done:
            # OPTIMIZATION: Clear and redraw efficiently
            ax.clear()
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.set_aspect('equal')
            ax.set_title(f"Method: {args.method} | Step: {step_count}")
            
            # Draw Static Walls
            for w in env.walls:
                rect = patches.Rectangle((w[0], w[2]), w[1]-w[0], w[3]-w[2], color='gray', alpha=0.8)
                ax.add_patch(rect)
                
            # Draw Entities
            ax.add_patch(patches.Circle((env.goal_pos[0], env.goal_pos[1]), env.goal_radius, color='gold', label='Exit'))
            ax.add_patch(patches.Circle((env.harry_pos[0], env.harry_pos[1]), 0.2, color='blue', label='Harry'))
            ax.add_patch(patches.Circle((env.filch_pos[0], env.filch_pos[1]), 0.2, color='brown', label='Filch'))
            ax.add_patch(patches.Circle((env.cat_pos[0], env.cat_pos[1]), 0.2, color='red', label='Mrs. Norris'))
            
            # Radii
            ax.add_patch(patches.Circle((env.cat_pos[0], env.cat_pos[1]), env.smell_radius, color='red', alpha=0.05))
            ax.add_patch(patches.Circle((env.harry_pos[0], env.harry_pos[1]), env.sight_radius, color='blue', alpha=0.05))
            
            # Refresh the GUI
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01) # Small pause to let the window update

            # 3. Model Inference
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                # We use the mean (action[0]) for smooth visualization
                action_data = agent.get_action(obs_tensor)
                action = action_data[0].cpu().squeeze().numpy()
            
            # 4. Environment Step
            obs, reward, done, _, info = env.step(action)
            step_count += 1
            
            if step_count > 1000: # Safety break
                break

    except KeyboardInterrupt:
        print("Visualization stopped by user.")

    plt.ioff()
    print(f"Final Result: {info.get('result', 'N/A')}")
    plt.show() # Keep the final frame open

if __name__ == '__main__':
    # Run with
    # python run.py --method ppo --checkpoint ppo/ppo_10k.pt
    # python run.py --method a2c_shared --checkpoint a2c/a2c_10k.pt
    # python run.py --method a2c_separate --checkpoint a2c_separate/a2c_separate_10k.pt
    main()