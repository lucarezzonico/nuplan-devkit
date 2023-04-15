import torch
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention_weights(attn_weights: torch.Tensor):
    """
    Plots the attention weights as a heatmap.

    Args:
        attn_weights (torch.Tensor): Tensor of shape [batch_size, target_sequence_length, source_sequence_length].
    """
    batch_size, target_sequence_length, source_sequence_length = attn_weights.size()

    # Plot the attention weights as a heatmap.
    fig, ax = plt.subplots(figsize=(90,6))
    sns.heatmap(attn_weights.squeeze(dim=1).cpu().detach().numpy(), cmap="YlGnBu", ax=ax)
    ax.set_xlabel("Source Sequence")
    ax.set_ylabel("Target Sequence")
    ax.set_title("Attention Weights each batch")
    plt.show()
        
if __name__ == '__main__':
    tensor = torch.rand(16, 1, 191)
    plot_attention_weights(tensor)