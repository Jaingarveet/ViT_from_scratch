import torch
from torch import nn
import torchinfo
from torchinfo import summary

class PatchEmbeddings(nn.Module):

  def __init__(self,
               hidden_dimension:int,
               patch_resolution:int,
               in_channels:int):

    super().__init__()
    # Use a CNN over this, NCHW-N,768,14,14 (for original patch_res=16,channels=3)
    self.create_patches_and_project = nn.Conv2d(in_channels=in_channels,
                                    out_channels= hidden_dimension,
                                    kernel_size=(patch_resolution,patch_resolution),
                                    stride=patch_resolution,
                                    dilation=1)

    # self.flattened_patch_size = int(in_channels * (patch_resolution**2))
    self.flatten_feature_map = nn.Flatten(start_dim=-2, end_dim=-1)
    # self.project_patches = nn.Linear(in_features=self.flattened_patch_size,
    #                                      out_features=hidden_dimension)

  def forward(self,x) -> torch.Tensor:
    x = self.create_patches_and_project(x)
    x = self.flatten_feature_map(x)
    x = x.transpose(-2,-1)
    return x
    # flatten -> linearly project



# N, num_patches, hidden_size
class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads:int, embed_size: int):
    super().__init__()
    self.head_dimension = int(embed_size / num_heads)
    self.num_heads = num_heads
    # self.attention_heads = nn.ModuleList([SelfAttention(hidden_size = embed_size,
                                                # KQV_projection_dim = self.head_dimension) for i in range(num_heads)])
    
    self.final_linear_post_concat = nn.Linear(in_features=embed_size,out_features=embed_size)

    self.layer_norm = nn.LayerNorm(normalized_shape=embed_size)
    self.QKV_linear = nn.Linear(in_features= embed_size,
                                 out_features= int(3 * embed_size))

  def forward(self,x) -> torch.Tensor:
      x = self.layer_norm(x)
      N,num_patches, hidden_size = x.shape
      # N, num_patches, hidden_size ->  N, num_patches, 3 * hidden_size
      qkv_concat = self.QKV_linear(x)
      qkv_concat = qkv_concat.reshape(N, num_patches, 3, self.num_heads, self.head_dimension)

      Q,K,V = qkv_concat.permute(2,0,3,1,4)
      # (3,N,num_heads,num_patches,head_dim)

      K_t = K.transpose(-2,-1)
      attention = torch.matmul(Q,K_t)
      scaled_attention = attention / (self.head_dimension ** 0.5)
      attention_weights = torch.softmax(scaled_attention,dim=-1) # NxN
      weighted_values = torch.matmul(attention_weights,V)
      # (N,num_heads,num_patches,head_dim)
      concat_output = weighted_values.transpose(-3,-2).reshape(N,num_patches,hidden_size)

      # # N, num_patches, num_heads, head_dimension
      # pre_concat_output = []
      # for attention_block in self.attention_heads:
      #   pre_concat_output.append(attention_block(x))
      #   # check this
      # concat_output = torch.cat(pre_concat_output,dim=-1)
      final_output = self.final_linear_post_concat(concat_output)

      return final_output


# no dropout is mentioned for MSA and patch_embeddings in training in appendix b -> b.1
class MLPBlock(nn.Module):
  def __init__(self,embed_size:int,MLP_size:int,dropout_rate:float):
    super().__init__()

    self.layer_norm = nn.LayerNorm(normalized_shape=embed_size)

    self.MLP = nn.Sequential(
        nn.Linear(in_features=embed_size,out_features=MLP_size),
        nn.GELU(),
        nn.Dropout(p=dropout_rate), # from table 3
        nn.Linear(in_features=MLP_size,out_features=embed_size),
        nn.Dropout(p=dropout_rate))

  def forward(self,x) -> torch.Tensor:
    post_MLP_output = self.MLP(self.layer_norm(x))
    return post_MLP_output

class TransformerEncoder(nn.Module):
  def __init__(self, hidden_size:int,
               MLP_size:int,
               num_heads:int,
               dropout_rate:int):

    super().__init__()

    self.MSA = MultiHeadAttention(num_heads = num_heads,
                                  embed_size = hidden_size)

    self.MLP = MLPBlock(embed_size = hidden_size,
                        MLP_size=MLP_size,
                        dropout_rate=dropout_rate)

  def forward(self,x) -> torch.Tensor:
    # N, num_patches, patch_embedding_projection_out_features .....dimensions of x -> N, num_patches, patch_embedding_projection_out_features
    msa_residual_output = self.MSA(x) + x
    # N, num_patches, patch_embedding_projection_out_features
    mlp_residual_output = self.MLP(msa_residual_output) + msa_residual_output

    return mlp_residual_output

class ViT(nn.Module):
  def __init__(self,
               patch_projection_size:int,
               patch_resolution:int,
               num_patches:int,
               in_channels:int,
               num_transformer_layers:int,
               dropout_rate:int,
               num_heads:int,
               MLP_size:int,
               num_classes:int
               ):
    # Change the parameter and see once how it is working.
    # number of tansformer encoder layers/blocks in base model ViT is 12
    super().__init__()

    self.patch_embedding = PatchEmbeddings(in_channels= in_channels,
                                           hidden_dimension=patch_projection_size,
                                           patch_resolution=patch_resolution)

    self.class_token = nn.Parameter(torch.randn(1,1,patch_projection_size) * 0.02) 

    self.position_embeddings = nn.Parameter(torch.randn(1,num_patches+1,
                                                   patch_projection_size)* 0.02)

    self.transformer_layers = nn.ModuleList([TransformerEncoder(hidden_size=patch_projection_size,
                                                                MLP_size=MLP_size,
                                                                num_heads=num_heads,
                                                                # output dimension of patches = input dimension of MSA
                                                                dropout_rate=dropout_rate)
                                                                for _ in range(num_transformer_layers)])

    self.transformer_sequence = nn.Sequential(*self.transformer_layers)

    self.MLP_head = nn.Sequential(
        nn.LayerNorm(normalized_shape = patch_projection_size),
        nn.Linear(in_features = patch_projection_size,
                  out_features = num_classes)
    )

  def forward(self, x) -> torch.Tensor:
    # N = BATCH_SIZE
    # NCHW -> N,num_patches,P**2.C -> N,num_patches,patch_projection_size
    batch_size = x.shape[0]
    flattened_patch_embeddings = self.patch_embedding(x)
    # check the flattened patch_embeddings here
    # N, num_patches, patch_projection_size -> N, num_patches, patch_projection_size
    class_embed = self.class_token.expand(batch_size,-1,-1)

    position_embeddings = self.position_embeddings.expand(batch_size,-1,-1)
    
    position_and_patch_embeddings = torch.cat((class_embed,
                                               flattened_patch_embeddings),dim=1) + position_embeddings

    # N, num_patches, patch_projection_size -> N, num_patches, patch_projection_size
    pre_MLP_HEAD_output = self.transformer_sequence(position_and_patch_embeddings)

    # accessing all the very first tokens and it's embeddings to get class_token_embeddings through the batch
    pred_logits = self.MLP_head(pre_MLP_HEAD_output[:,0])

    return pred_logits