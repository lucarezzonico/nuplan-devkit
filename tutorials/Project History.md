# TODO



# LaneGCN Model Structure

```
LightningModuleWrapper(
  (model): LaneGCN(
    (lane_net): LaneNet(
      (input): Sequential(
        (0): Linear(in_features=2, out_features=128, bias=True)
        (1): ReLU(inplace=True)
        (2): LinearWithGroupNorm(
          (linear): Linear(in_features=128, out_features=128, bias=False)
          (norm): GroupNorm(1, 128, eps=1e-05, affine=True)
          (relu): ReLU(inplace=True)
        )
      )
      (_seg): Sequential(
        (0): Linear(in_features=2, out_features=128, bias=True)
        (1): ReLU(inplace=True)
        (2): LinearWithGroupNorm(
          (linear): Linear(in_features=128, out_features=128, bias=False)
          (norm): GroupNorm(1, 128, eps=1e-05, affine=True)
          (relu): ReLU(inplace=True)
        )
      )
      (_relu): ReLU(inplace=True)
      (fusion_net): ModuleDict(
        (center): ModuleList(
          (0): Linear(in_features=128, out_features=128, bias=False)
          (1): Linear(in_features=128, out_features=128, bias=False)
          (2): Linear(in_features=128, out_features=128, bias=False)
        )
        (group_norm): ModuleList(
          (0): GroupNorm(1, 128, eps=1e-05, affine=True)
          (1): GroupNorm(1, 128, eps=1e-05, affine=True)
          (2): GroupNorm(1, 128, eps=1e-05, affine=True)
        )
        (linear_w_group_norm): ModuleList(
          (0): LinearWithGroupNorm(
            (linear): Linear(in_features=128, out_features=128, bias=False)
            (norm): GroupNorm(1, 128, eps=1e-05, affine=True)
            (relu): ReLU(inplace=True)
          )
          (1): LinearWithGroupNorm(
            (linear): Linear(in_features=128, out_features=128, bias=False)
            (norm): GroupNorm(1, 128, eps=1e-05, affine=True)
            (relu): ReLU(inplace=True)
          )
          (2): LinearWithGroupNorm(
            (linear): Linear(in_features=128, out_features=128, bias=False)
            (norm): GroupNorm(1, 128, eps=1e-05, affine=True)
            (relu): ReLU(inplace=True)
          )
        )
        (pre1): ModuleList(
          (0): Linear(in_features=128, out_features=128, bias=False)
          (1): Linear(in_features=128, out_features=128, bias=False)
          (2): Linear(in_features=128, out_features=128, bias=False)
        )
        (suc1): ModuleList(
          (0): Linear(in_features=128, out_features=128, bias=False)
          (1): Linear(in_features=128, out_features=128, bias=False)
          (2): Linear(in_features=128, out_features=128, bias=False)
        )
        (pre2): ModuleList(
          (0): Linear(in_features=128, out_features=128, bias=False)
          (1): Linear(in_features=128, out_features=128, bias=False)
          (2): Linear(in_features=128, out_features=128, bias=False)
        )
        (suc2): ModuleList(
          (0): Linear(in_features=128, out_features=128, bias=False)
          (1): Linear(in_features=128, out_features=128, bias=False)
          (2): Linear(in_features=128, out_features=128, bias=False)
        )
        (pre3): ModuleList(
          (0): Linear(in_features=128, out_features=128, bias=False)
          (1): Linear(in_features=128, out_features=128, bias=False)
          (2): Linear(in_features=128, out_features=128, bias=False)
        )
        (suc3): ModuleList(
          (0): Linear(in_features=128, out_features=128, bias=False)
          (1): Linear(in_features=128, out_features=128, bias=False)
          (2): Linear(in_features=128, out_features=128, bias=False)
        )
        (pre4): ModuleList(
          (0): Linear(in_features=128, out_features=128, bias=False)
          (1): Linear(in_features=128, out_features=128, bias=False)
          (2): Linear(in_features=128, out_features=128, bias=False)
        )
        (suc4): ModuleList(
          (0): Linear(in_features=128, out_features=128, bias=False)
          (1): Linear(in_features=128, out_features=128, bias=False)
          (2): Linear(in_features=128, out_features=128, bias=False)
        )
      )
    )
    (ego_feature_extractor): Sequential(
      (0): Linear(in_features=15, out_features=128, bias=True)
      (1): ReLU(inplace=True)
      (2): Linear(in_features=128, out_features=128, bias=True)
      (3): ReLU(inplace=True)
      (4): LinearWithGroupNorm(
        (linear): Linear(in_features=128, out_features=128, bias=False)
        (norm): GroupNorm(1, 128, eps=1e-05, affine=True)
        (relu): ReLU(inplace=True)
      )
    )
    (agent_feature_extractor): Sequential(
      (0): Linear(in_features=40, out_features=128, bias=True)
      (1): ReLU(inplace=True)
      (2): Linear(in_features=128, out_features=128, bias=True)
      (3): ReLU()
      (4): LinearWithGroupNorm(
        (linear): Linear(in_features=128, out_features=128, bias=False)
        (norm): GroupNorm(1, 128, eps=1e-05, affine=True)
        (relu): ReLU(inplace=True)
      )
    )
    (actor2lane_attention): Actor2LaneAttention(
      (lane_meta): LinearWithGroupNorm(
        (linear): Linear(in_features=134, out_features=128, bias=False)
        (norm): GroupNorm(1, 128, eps=1e-05, affine=True)
        (relu): ReLU(inplace=True)
      )
      (attention_layers): ModuleList(
        (0): GraphAttention(
          (src_encoder): Sequential(
            (0): Linear(in_features=128, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (dst_encoder): Sequential(
            (0): Linear(in_features=128, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (edge_dist_encoder): Sequential(
            (0): Linear(in_features=2, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (edge_encoder): Sequential(
            (0): Linear(in_features=384, out_features=128, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=128, out_features=128, bias=True)
          )
          (dst_feature_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (output_linear): Linear(in_features=128, out_features=128, bias=True)
        )
        (1): GraphAttention(
          (src_encoder): Sequential(
            (0): Linear(in_features=128, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (dst_encoder): Sequential(
            (0): Linear(in_features=128, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (edge_dist_encoder): Sequential(
            (0): Linear(in_features=2, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (edge_encoder): Sequential(
            (0): Linear(in_features=384, out_features=128, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=128, out_features=128, bias=True)
          )
          (dst_feature_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (output_linear): Linear(in_features=128, out_features=128, bias=True)
        )
        (2): GraphAttention(
          (src_encoder): Sequential(
            (0): Linear(in_features=128, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (dst_encoder): Sequential(
            (0): Linear(in_features=128, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (edge_dist_encoder): Sequential(
            (0): Linear(in_features=2, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (edge_encoder): Sequential(
            (0): Linear(in_features=384, out_features=128, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=128, out_features=128, bias=True)
          )
          (dst_feature_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (output_linear): Linear(in_features=128, out_features=128, bias=True)
        )
        (3): GraphAttention(
          (src_encoder): Sequential(
            (0): Linear(in_features=128, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (dst_encoder): Sequential(
            (0): Linear(in_features=128, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (edge_dist_encoder): Sequential(
            (0): Linear(in_features=2, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (edge_encoder): Sequential(
            (0): Linear(in_features=384, out_features=128, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=128, out_features=128, bias=True)
          )
          (dst_feature_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (output_linear): Linear(in_features=128, out_features=128, bias=True)
        )
      )
    )
    (lane2actor_attention): Lane2ActorAttention(
      (attention_layers): ModuleList(
        (0): GraphAttention(
          (src_encoder): Sequential(
            (0): Linear(in_features=128, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (dst_encoder): Sequential(
            (0): Linear(in_features=128, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (edge_dist_encoder): Sequential(
            (0): Linear(in_features=2, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (edge_encoder): Sequential(
            (0): Linear(in_features=384, out_features=128, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=128, out_features=128, bias=True)
          )
          (dst_feature_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (output_linear): Linear(in_features=128, out_features=128, bias=True)
        )
        (1): GraphAttention(
          (src_encoder): Sequential(
            (0): Linear(in_features=128, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (dst_encoder): Sequential(
            (0): Linear(in_features=128, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (edge_dist_encoder): Sequential(
            (0): Linear(in_features=2, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (edge_encoder): Sequential(
            (0): Linear(in_features=384, out_features=128, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=128, out_features=128, bias=True)
          )
          (dst_feature_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (output_linear): Linear(in_features=128, out_features=128, bias=True)
        )
        (2): GraphAttention(
          (src_encoder): Sequential(
            (0): Linear(in_features=128, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (dst_encoder): Sequential(
            (0): Linear(in_features=128, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (edge_dist_encoder): Sequential(
            (0): Linear(in_features=2, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (edge_encoder): Sequential(
            (0): Linear(in_features=384, out_features=128, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=128, out_features=128, bias=True)
          )
          (dst_feature_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (output_linear): Linear(in_features=128, out_features=128, bias=True)
        )
        (3): GraphAttention(
          (src_encoder): Sequential(
            (0): Linear(in_features=128, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (dst_encoder): Sequential(
            (0): Linear(in_features=128, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (edge_dist_encoder): Sequential(
            (0): Linear(in_features=2, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (edge_encoder): Sequential(
            (0): Linear(in_features=384, out_features=128, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=128, out_features=128, bias=True)
          )
          (dst_feature_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (output_linear): Linear(in_features=128, out_features=128, bias=True)
        )
      )
    )
    (actor2actor_attention): Actor2ActorAttention(
      (attention_layers): ModuleList(
        (0): GraphAttention(
          (src_encoder): Sequential(
            (0): Linear(in_features=128, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (dst_encoder): Sequential(
            (0): Linear(in_features=128, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (edge_dist_encoder): Sequential(
            (0): Linear(in_features=2, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (edge_encoder): Sequential(
            (0): Linear(in_features=384, out_features=128, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=128, out_features=128, bias=True)
          )
          (dst_feature_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (output_linear): Linear(in_features=128, out_features=128, bias=True)
        )
        (1): GraphAttention(
          (src_encoder): Sequential(
            (0): Linear(in_features=128, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (dst_encoder): Sequential(
            (0): Linear(in_features=128, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (edge_dist_encoder): Sequential(
            (0): Linear(in_features=2, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (edge_encoder): Sequential(
            (0): Linear(in_features=384, out_features=128, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=128, out_features=128, bias=True)
          )
          (dst_feature_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (output_linear): Linear(in_features=128, out_features=128, bias=True)
        )
        (2): GraphAttention(
          (src_encoder): Sequential(
            (0): Linear(in_features=128, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (dst_encoder): Sequential(
            (0): Linear(in_features=128, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (edge_dist_encoder): Sequential(
            (0): Linear(in_features=2, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (edge_encoder): Sequential(
            (0): Linear(in_features=384, out_features=128, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=128, out_features=128, bias=True)
          )
          (dst_feature_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (output_linear): Linear(in_features=128, out_features=128, bias=True)
        )
        (3): GraphAttention(
          (src_encoder): Sequential(
            (0): Linear(in_features=128, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (dst_encoder): Sequential(
            (0): Linear(in_features=128, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (edge_dist_encoder): Sequential(
            (0): Linear(in_features=2, out_features=128, bias=True)
            (1): ReLU(inplace=True)
          )
          (edge_encoder): Sequential(
            (0): Linear(in_features=384, out_features=128, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=128, out_features=128, bias=True)
          )
          (dst_feature_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (output_linear): Linear(in_features=128, out_features=128, bias=True)
        )
      )
    )
    (_mlp): Sequential(
      (0): Linear(in_features=128, out_features=128, bias=True)
      (1): ReLU()
      (2): Linear(in_features=128, out_features=128, bias=True)
      (3): ReLU()
      (4): Linear(in_features=128, out_features=48, bias=True)
    )
  )
)
```
