import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

def visualize_episode(env, model):
    obs, _ = env.reset()
    done = False
    
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
            
        # Draw Goal
        goal = patches.Circle((env.goal_pos[0], env.goal_pos[1]), env.goal_radius, color='gold', label='Exit')
        ax.add_patch(goal)
        
        # Draw Entities
        harry = patches.Circle((env.harry_pos[0], env.harry_pos[1]), 0.2, color='blue', label='Harry')
        filch = patches.Circle((env.filch_pos[0], env.filch_pos[1]), 0.2, color='brown', label='Filch')
        cat = patches.Circle((env.cat_pos[0], env.cat_pos[1]), 0.2, color='red', label='Mrs. Norris')
        
        # Smell radius representation
        cat_smell = patches.Circle((env.cat_pos[0], env.cat_pos[1]), env.smell_radius, color='red', alpha=0.1)
        # Sight radius representation
        harry_sight = patches.Circle((env.harry_pos[0], env.harry_pos[1]), env.sight_radius, color='blue', alpha=0.1)
        
        ax.add_patch(harry)
        ax.add_patch(filch)
        ax.add_patch(cat)
        ax.add_patch(cat_smell)
        ax.add_patch(harry_sight)
        
        ax.legend(loc='upper right')
        plt.pause(0.05)
        
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = model.get_action(obs_tensor)
            
        obs, _, done, _, _ = env.step(action.squeeze().numpy())
        
    plt.ioff()
    plt.show()

# if __name__ == "__main__":
    # --- RUN SCRIPT ---
    # env = HarryPotterEnv()
    # model = ActorCriticNet(10, 2)
    # print("Training PPO...")
    # train_ppo(env, model, episodes=2000)
    # visualize_episode(env, model)