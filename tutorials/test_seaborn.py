import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

def plot_attention_weights(attn_weights: torch.Tensor):
    """
    Plots the attention weights as a heatmap.

    Args:
        attn_weights (torch.Tensor): Tensor of shape [batch_size, target_sequence_length, source_sequence_length].
    """
    batch_size, target_sequence_length, source_sequence_length = attn_weights.size()

    # Plot the attention weights as a heatmap.
    fig, axs = plt.subplots(nrows=batch_size, figsize=(1, 5*batch_size))
    for i in range(batch_size):
        sns.heatmap(attn_weights[i].cpu().detach().numpy(), cmap="YlGnBu", ax=axs[i])
        axs[i].set_xlabel("Source Sequence")
        axs[i].set_ylabel("Target Sequence")
        axs[i].set_title(f"Attention Weights for Example {i+1}")
    plt.show()
        
if __name__ == '__main__':
    tensor = torch.rand(16, 1, 191)
    plot_attention_weights(tensor)