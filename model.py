import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math




class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SimpleAdapter(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features           
        hidden_features = hidden_features or in_features     
        self.fc1 = nn.Linear(in_features, hidden_features)       # Downconv
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)      # UPconv
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)            
        x = self.act(x)            
        x = self.drop(x)
        x = self.fc2(x)           
        x = self.drop(x)   
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x_norm1 = self.norm1(x).contiguous().clone()
        res = self.drop_path(self.attn(x_norm1, H, W))
        x = x + res  # residual add

        x_norm2 = self.norm2(x).contiguous().clone()
        res = self.drop_path(self.mlp(x_norm2, H, W))
        x = x + res

        return x


class MOEAdapterBlock(nn.Module):
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, 
                 expert_num=None, domain_num=None, select_mode=None, MoE_hidden_dim=None, num_k = None):   
        
        super().__init__()
        
        self.expert_num = expert_num
        self.domain_num = domain_num
        self.select_mode = select_mode
        self.acc_freq = 0 
        self.num_K = num_k
        self.MI_task_gate = torch.zeros(self.domain_num, self.expert_num).cuda()    

        self.norm1 = norm_layer(dim) 
        self.norm2 = norm_layer(dim)
        
        self.attn = Attention(
            dim,
            num_heads=num_heads,     
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale,
            attn_drop=attn_drop, 
            proj_drop=drop, 
            sr_ratio=sr_ratio)
        
        mlp_hidden_dim = int(dim * mlp_ratio)      
        
        self.mlp = Mlp(in_features=dim,    
                       hidden_features=mlp_hidden_dim,   
                       act_layer=act_layer, 
                       drop=drop)
        
        if self.select_mode == 'new_topk':      # NOTE you can add other routing methods 
            self.softplus = nn.Softplus()
            self.softmax = nn.Softmax(1)
            self.f_gate = nn.ModuleList([nn.Sequential(nn.Linear(dim, 2 * expert_num, bias=False)) for i in range(self.domain_num)])   
            
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()   
        self.apply(self._init_weights)    
        
        expert_lists = []
        for _ in range(expert_num) : 
            tmp_adapter = SimpleAdapter(in_features=dim, hidden_features=MoE_hidden_dim, act_layer=act_layer, drop=drop)  
            tmp_adapter.apply(self._init_weights)      
            expert_lists.append(tmp_adapter)

        self.adapter_experts = nn.ModuleList(expert_lists)   

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)   
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def minmax_scaling(self, top_k_logits) : 
            m1 = top_k_logits.min() ; m2 = top_k_logits.max()
            return (top_k_logits-m1)/ (m2-m1)
            
    def one_hot_encoding(self, index, num_classes):
        one_hot = np.zeros(num_classes)  
        one_hot[index] = 1  
        return one_hot

    def forward(self, x, H, W, expert_num, select_mode, pseudo_domain_label=None, expert_check=False):    
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))  
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))   

        self.MI_task_gate = torch.zeros(self.domain_num, self.expert_num)  # device 고려

        if self.select_mode == 'random':    
            select = torch.randint(low=0, high=expert_num, size=(1,)).item()  
            MI_loss = self.one_hot_encoding(select, self.expert_num)     
            x = x + self.adapter_experts[select](x, H, W)    

        elif self.select_mode == 'new_topk':
            task_bh = pseudo_domain_label.tolist()

            # (B, 2E)
            total_w = torch.stack([
                self.f_gate[task_bh[i]](x[i]) for i in range(x.size(0))
            ], dim=0)

            clean_logits, raw_noise_stddev = total_w.chunk(2, dim=-1)
            noise_stddev = F.softplus(raw_noise_stddev) + 1e-2
            logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev

            exp_wise_sum = logits.sum(dim=1)  # (B, E)
            probs = F.softmax(exp_wise_sum, dim=-1)

            for i, t in enumerate(task_bh):
                self.MI_task_gate[t] += probs[i].detach().cpu()

            top_k = min(self.num_K + 1, self.expert_num)
            top_logits, top_indices = exp_wise_sum.topk(top_k, dim=1)
            top_k_logits = top_logits[:, :self.num_K]      # [B, K]
            top_k_indices = top_indices[:, :self.num_K]    # [B, K]

            if top_k_logits.size(0) > 1:
                top_k_gates = self.softmax(self.minmax_scaling(top_k_logits))  # [B, K]
            else:
                top_k_gates = self.softmax(top_k_logits)

            # Adapter output 계산 (B, D)
            adapter_outputs = torch.stack(
                [self.adapter_experts[e](x, H, W) for e in range(self.expert_num)], dim=1
            )

            B, S, D = x.shape
            K = self.num_K
            top_k_exp_outputs = torch.zeros(B, K, S, D, device=x.device)

            # 각 expert만 따로 호출 (loop는 있지만 실행 expert만 돌기 때문에 메모리 효율 ↑)
            for k in range(K):
                selected_expert_ids = top_k_indices[:, k]  # (B,)
                x_k = []

                for b in range(B):
                    e_id = selected_expert_ids[b].item()
                    x_b = x[b].unsqueeze(0)  # (1, S, D)
                    out = self.adapter_experts[e_id](x_b, H, W)  # (1, S, D)
                    x_k.append(out)

                # Stack across batch: (B, S, D)
                x_k = torch.cat(x_k, dim=0)
                top_k_exp_outputs[:, k] = x_k

            # apply gates: (B, K, 1, 1)
            gates = top_k_gates.unsqueeze(-1).unsqueeze(-1)
            weighted_outputs = gates * top_k_exp_outputs  # (B, K, S, D)

            # sum over experts: (B, S, D)
            x_out = weighted_outputs.sum(dim=1)

            x = x + x_out

            self.MI_task_gate = self.MI_task_gate / x.size(0)

            
            P_TI = torch.sum(self.MI_task_gate, dim=1, keepdim=True) + 1e-4
            P_EI = torch.sum(self.MI_task_gate, dim=0, keepdim=True) + 1e-4

            MI_loss = ((self.MI_task_gate + 1e-4).cuda() * torch.log(self.MI_task_gate.cuda() / (P_TI * P_EI).cuda() + 1e-4)).sum()
            
            if expert_check:
                return x, MI_loss, top_k_indices
        else:
            print('No attribute')
            MI_loss = None
         
        return x, MI_loss  


class MOEAdapterBlockR(nn.Module):
    def __init__(
        self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
        drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
        sr_ratio=1, expert_num=None, num_domains=None, select_mode=None,
        MoE_hidden_dim=None, num_k=None
    ):
        super().__init__()
        self.expert_num = expert_num
        self.domain_num = num_domains
        self.select_mode = select_mode
        self.num_K = num_k
    
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
    
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio
        )
    
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
        if self.select_mode == 'new_topk':
            self.softplus = nn.Softplus()
            self.softmax = nn.Softmax(dim=-1)
            self.f_gate = nn.ModuleList([
                nn.Sequential(nn.Linear(dim, 2 * expert_num, bias=False))
                for _ in range(self.domain_num)
            ])
    
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)
    
        expert_lists = []
        for _ in range(expert_num):
            tmp_adapter = SimpleAdapter(in_features=dim, hidden_features=MoE_hidden_dim, act_layer=act_layer, drop=drop)
            tmp_adapter.apply(self._init_weights)
            expert_lists.append(tmp_adapter)
        self.adapter_experts = nn.ModuleList(expert_lists)
    
    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def minmax_scaling(self, top_k_logits):
        m1 = top_k_logits.min()
        m2 = top_k_logits.max()
        return (top_k_logits - m1) / (m2 - m1)
    
    def forward(self, x, H, W, expert_num, select_mode, pseudo_domain_label, R):
        """
        x: (B, N, C)    - 토큰 시퀀스
        R: (D, D) or None - 도메인 간 유사도 행렬 (None이면 relational term 건너뜀)
        pseudo_domain_label: (B,) 각 샘플의 도메인 인덱스
        """
        B, N, C = x.shape
    
        # 1) Transformer Flow: Attention + MLP
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
    
        # 2) MI용 joint 분포 누적
        MI_task_gate = torch.zeros(self.domain_num, self.expert_num, device=x.device)
    
        # 3) routing / expert 어댑터 적용
        if select_mode == 'random':
            select = torch.randint(low=0, high=expert_num, size=(1,), device=x.device).item()
            MI_loss = 0.0
            x = x + self.adapter_experts[select](x, H, W)
            return x, MI_loss
    
        elif select_mode == 'new_topk':
            # -- (a) 각 샘플별 토큰 수준에서 로짓 생성 및 합산 --
            per_sample_token_logits = []
            for i in range(B):
                gate_out = self.f_gate[pseudo_domain_label[i].item()](x[i])  # (N, 2E)
                clean_logits, raw_noise = gate_out.chunk(2, dim=-1)          # (N, E)
                noise_std = F.softplus(raw_noise) + 1e-2                       # (N, E)
                logits_i = clean_logits + torch.randn_like(clean_logits) * noise_std  # (N, E)
                per_sample_token_logits.append(logits_i)                      # 리스트에 (N, E)
    
            # -- (b) 토큰 축(axis=0) 합산 → (B, E) 벡터 획득 --
            token_logits_sum = torch.stack([
                logits_i.sum(dim=0)  # (E,)
                for logits_i in per_sample_token_logits
            ], dim=0)  # shape = (B, E)
    
            # -- (c) softmax → 샘플별 expert 확률 (B, E) --
            probs = F.softmax(token_logits_sum, dim=-1)  # (B, E)
    
            # -- (d) joint P(D, E) 누적 --
            for i, d in enumerate(pseudo_domain_label):
                MI_task_gate[d] += probs[i].detach()  # probs[i]: (E,)
    
            # -- (e) Top-K expert indices 및 게이트 점수 --
            top_k = min(self.num_K + 1, self.expert_num)
            top_logits, top_indices = token_logits_sum.topk(top_k, dim=-1)  # (B, top_k)
            top_k_logits = top_logits[:, :self.num_K]    # (B, K)
            top_k_indices = top_indices[:, :self.num_K]  # (B, K)
    
            if top_k_logits.size(0) > 1:
                top_k_gates = F.softmax(self.minmax_scaling(top_k_logits), dim=-1)  # (B, K)
            else:
                top_k_gates = F.softmax(top_k_logits, dim=-1).unsqueeze(-1)         # (1, K)
    
            # -- (f) Adapter expert 출력 계산 (메모리 절약용 루프) --
            K = self.num_K
            top_k_exp_outputs = torch.zeros(B, K, N, C, device=x.device)
    
            for k_idx in range(K):
                sel_ids = top_k_indices[:, k_idx]  # (B,)
                outs_k = []
                for b in range(B):
                    e_id = sel_ids[b].item()        # 예: 3
                    xb = x[b].unsqueeze(0)           # (1, N, C)
                    out_b = self.adapter_experts[e_id](xb, H, W)  # (1, N, C)
                    outs_k.append(out_b)
                outs_k = torch.cat(outs_k, dim=0)  # (B, N, C)
                top_k_exp_outputs[:, k_idx] = outs_k
    
            gates = top_k_gates.unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1)
            weighted_outputs = gates * top_k_exp_outputs      # (B, K, N, C)
            x_out = weighted_outputs.sum(dim=1)               # (B, N, C)
            x = x + x_out
    
            # -- (g) joint P(D, E) 정규화 → MI 계산 --
            MI_task_gate = MI_task_gate / B  # (D, E)
    
            P_D = MI_task_gate.sum(dim=1, keepdim=True) + 1e-6   # (D, 1)
            P_E = MI_task_gate.sum(dim=0, keepdim=True) + 1e-6   # (1, E)
            MI = (MI_task_gate * torch.log(MI_task_gate / (P_D * P_E) + 1e-6)).sum()
    
            # -- (h) P(E | D) 분포 계산 --
            P_E_given_D = MI_task_gate / P_D  # (D, E)
    
            # -- (i) R이 주어진 경우에만 relational penalty 계산 --
            if R is not None:
                D_dim, E_dim = P_E_given_D.shape
                rel_term = 0.0
                for i in range(D_dim):
                    for j in range(D_dim):
                        diff = (P_E_given_D[i] - P_E_given_D[j]).pow(2).sum()
                        rel_term += R[i, j] * diff
                rel_term = rel_term / (D_dim * D_dim)
                MI_loss = MI - 1e-1*rel_term
            else:
                # R이 None이면 relational term 생략 → 단순히 -MI만 사용
                MI_loss = MI
    
            return x, MI_loss
    
        else:
            # select_mode이 없거나 다른 모드라면 MI_loss=0 반환
            return x, torch.zeros(1, device=x.device)

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out_1 = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out = fan_out_1 // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x_2 = self.norm(x)
        return x_2.clone(), H, W


class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=2, embed_dims=[32, 64, 128, 256], 
                 dims=(32, 64, 160, 256), decoder_dim=256, 
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])


        

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.to_fused = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, decoder_dim, kernel_size=1),
                nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)
            ) for i, dim in enumerate(dims)
        ])

        # to_segmentation: 두 번의 1x1 Conv
        self.head = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, kernel_size=1),
            nn.Conv2d(decoder_dim, num_classes, kernel_size=1)
        )


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        layer_outputs = self.forward_features(x)
        
        fused = [block(output) for output, block in zip(layer_outputs, self.to_fused)]
        fused = torch.cat(fused, dim=1)
        return self.head(fused)


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x




class MOE_MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=2,
                 embed_dims=[64, 128, 256, 512], dims=(32, 64, 160, 256),
                 decoder_dim=256, num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1], expert_num=12, select_mode=None, hidden_dims=None,
                 num_k=None, num_domains=12, expert_check=False):
        super().__init__()
        self.select_mode = select_mode
        self.expert_num = expert_num
        self.num_classes = num_classes
        self.num_k = num_k
        self.depths = depths
        self.num_domains = num_domains

        # Patch embeddings
        self.patch_embed1 = OverlapPatchEmbed(img_size, 7, 4, in_chans, embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size//4, 3, 2, embed_dims[0], embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size//8, 3, 2, embed_dims[1], embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size//16, 3, 2, embed_dims[2], embed_dims[3])

        # MOE blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.block1 = nn.ModuleList([
            MOEAdapterBlock(embed_dims[0], num_heads[0], mlp_ratios[0], qkv_bias, qk_scale,
                             drop_rate, attn_drop_rate, dpr[cur+i], nn.GELU, norm_layer, sr_ratios[0],
                             expert_num, num_domains, select_mode, hidden_dims[0], num_k)
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        cur += depths[0]

        self.block2 = nn.ModuleList([
            MOEAdapterBlock(embed_dims[1], num_heads[1], mlp_ratios[1], qkv_bias, qk_scale,
                             drop_rate, attn_drop_rate, dpr[cur+i], nn.GELU, norm_layer, sr_ratios[1],
                             expert_num, num_domains,  select_mode, hidden_dims[1], num_k)
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
        cur += depths[1]

        self.block3 = nn.ModuleList([
            MOEAdapterBlock(embed_dims[2], num_heads[2], mlp_ratios[2], qkv_bias, qk_scale,
                             drop_rate, attn_drop_rate, dpr[cur+i], nn.GELU, norm_layer, sr_ratios[2],
                             expert_num, num_domains, select_mode, hidden_dims[2], num_k)
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])
        cur += depths[2]

        self.block4 = nn.ModuleList([
            MOEAdapterBlock(embed_dims[3], num_heads[3], mlp_ratios[3], qkv_bias, qk_scale,
                             drop_rate, attn_drop_rate, dpr[cur+i], nn.GELU, norm_layer, sr_ratios[3],
                             expert_num, num_domains, select_mode, hidden_dims[3], num_k)
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # Decoder and segmentation head
        self.to_fused = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, decoder_dim, 1),
                nn.Upsample(size=(256,256), mode='bilinear', align_corners=False)
            ) for dim in dims])
        self.head = nn.Sequential(
            nn.Conv2d(4*decoder_dim, decoder_dim, 1),
            nn.Conv2d(decoder_dim, num_classes, 1)
        )

        # Domain classifier on pre-MoE features
        self.domain_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B,C,1,1)
            nn.Flatten(),             # (B,C)
            nn.Linear(embed_dims[3], num_domains)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        # ... (same init as before) ...
        pass

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()

    # def init_weights(self, pretrained=None):
    #     if isinstance(pretrained, str):
    #         logger = get_root_logger()
    #         load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    # freeze
    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def forward_features(self, x, pseudo_domain_label, expert_check=False):
        B = x.size(0)
        outs = []
        all_selected_experts = [
            [[] for _ in range(depth)]  # block1: 3개, block2: 6개, ...
            for depth in self.depths
        ]

        
        # --- Stage 1 ---
        x, H, W = self.patch_embed1(x)
        for blk_idx, blk in enumerate(self.block1):
            if expert_check:
                x, mi, sel_idx = blk(x, H, W, self.expert_num, self.select_mode, pseudo_domain_label, expert_check)
                all_selected_experts[0][blk_idx].append(sel_idx)
            else:
                x, mi = blk(x, H, W, self.expert_num, self.select_mode, pseudo_domain_label)
        x = self.norm1(x)  # (B, N1, C1)
        feat1 = x.view(B, H, W, -1).permute(0,3,1,2).contiguous()  # (B, C1, H, W)
        outs.append(feat1)
    
        # --- Stage 2 ---
        x, H, W = self.patch_embed2(feat1)
        for blk_idx, blk in enumerate(self.block2):
            if expert_check:
                x, mi, sel_idx = blk(x, H, W, self.expert_num, self.select_mode, pseudo_domain_label, expert_check)
                all_selected_experts[1][blk_idx].append(sel_idx)
            else:
                x, mi = blk(x, H, W, self.expert_num, self.select_mode, pseudo_domain_label)
        x = self.norm2(x)
        feat2 = x.view(B, H, W, -1).permute(0,3,1,2).contiguous()
        outs.append(feat2)
    
        # --- Stage 3 ---
        x, H, W = self.patch_embed3(feat2)
        for blk_idx, blk in enumerate(self.block3):
            if expert_check:
                x, mi, sel_idx = blk(x, H, W, self.expert_num, self.select_mode, pseudo_domain_label, expert_check)
                all_selected_experts[2][blk_idx].append(sel_idx)
            else:
                x, mi = blk(x, H, W, self.expert_num, self.select_mode, pseudo_domain_label)
        x = self.norm3(x)
        feat3 = x.view(B, H, W, -1).permute(0,3,1,2).contiguous()
        outs.append(feat3)
    
        # --- Stage 4 pre-MoE (token-level norm) ---
        x, H, W = self.patch_embed4(feat3)      # x: (B, N4, C4)
        x = self.norm4(x)                       # apply LayerNorm over last dim C4
        pre_moe_feat = x.view(B, H, W, -1) \
                       .permute(0,3,1,2)        # (B, C4, H, W), for domain classifier
    
        # --- Stage 4 MoE adapters ---
        total_MI_loss = 0
        for blk_idx, blk in enumerate(self.block4):
            if expert_check:
                x, mi, sel_idx = blk(x, H, W, self.expert_num, self.select_mode, pseudo_domain_label, expert_check)
                all_selected_experts[3][blk_idx].append(sel_idx)
            else:
                x, mi = blk(x, H, W, self.expert_num, self.select_mode, pseudo_domain_label)
            total_MI_loss += mi
    
        x = self.norm4(x)  # again normalize after adapters
        feat4 = x.view(B, H, W, -1).permute(0,3,1,2).contiguous()
        outs.append(feat4)
        if expert_check:
            return outs, pre_moe_feat, total_MI_loss, all_selected_experts
        return outs, pre_moe_feat, total_MI_loss
    def forward(self, x, pseudo_domain_label, expert_check=False):
        if expert_check:
            layer_outs, pre_moe_feat, total_MI, all_selected_experts = self.forward_features(x, pseudo_domain_label, expert_check)
        else:
            layer_outs, pre_moe_feat, total_MI = self.forward_features(x, pseudo_domain_label)
        # segmentation
        fused = torch.cat([f(o) for f, o in zip(self.to_fused, layer_outs)], dim=1)
        seg_out = self.head(fused)
        # domain pred from pre-MoE feature
        dom_logits = self.domain_classifier(pre_moe_feat)
        
        if expert_check:
            return seg_out, dom_logits, total_MI, all_selected_experts
        return seg_out, dom_logits, total_MI



    


class MOE_MixVisionTransformerv2(nn.Module):
    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, num_classes=2,
        embed_dims=[64, 128, 256, 512], dims=(32, 64, 160, 256),
        decoder_dim=256, num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1], expert_num=12, select_mode=None, hidden_dims=None,
        num_k=None, num_domains=12
    ):
        super().__init__()
        self.select_mode = select_mode
        self.expert_num = expert_num
        self.num_classes = num_classes
        self.num_k = num_k
        self.depths = depths
        self.num_domains = num_domains

        # Patch embeddings
        self.patch_embed1 = OverlapPatchEmbed(img_size, 7, 4, in_chans, embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size // 4, 3, 2, embed_dims[0], embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size // 8, 3, 2, embed_dims[1], embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size // 16, 3, 2, embed_dims[2], embed_dims[3])

        # MOE blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.block1 = nn.ModuleList([
            MOEAdapterBlockR(
                embed_dims[0], num_heads[0], mlp_ratios[0], qkv_bias, qk_scale,
                drop_rate, attn_drop_rate, dpr[cur + i], nn.GELU, norm_layer, sr_ratios[0],
                expert_num, num_domains, select_mode, hidden_dims[0], num_k
            )
            for i in range(depths[0])
        ])
        self.norm1 = norm_layer(embed_dims[0])
        cur += depths[0]

        self.block2 = nn.ModuleList([
            MOEAdapterBlockR(
                embed_dims[1], num_heads[1], mlp_ratios[1], qkv_bias, qk_scale,
                drop_rate, attn_drop_rate, dpr[cur + i], nn.GELU, norm_layer, sr_ratios[1],
                expert_num, num_domains, select_mode, hidden_dims[1], num_k
            )
            for i in range(depths[1])
        ])
        self.norm2 = norm_layer(embed_dims[1])
        cur += depths[1]

        self.block3 = nn.ModuleList([
            MOEAdapterBlockR(
                embed_dims[2], num_heads[2], mlp_ratios[2], qkv_bias, qk_scale,
                drop_rate, attn_drop_rate, dpr[cur + i], nn.GELU, norm_layer, sr_ratios[2],
                expert_num, num_domains, select_mode, hidden_dims[2], num_k
            )
            for i in range(depths[2])
        ])
        self.norm3 = norm_layer(embed_dims[2])
        cur += depths[2]

        self.block4 = nn.ModuleList([
            MOEAdapterBlockR(
                embed_dims[3], num_heads[3], mlp_ratios[3], qkv_bias, qk_scale,
                drop_rate, attn_drop_rate, dpr[cur + i], nn.GELU, norm_layer, sr_ratios[3],
                expert_num, num_domains, select_mode, hidden_dims[3], num_k
            )
            for i in range(depths[3])
        ])
        self.norm4 = norm_layer(embed_dims[3])

        # Decoder and segmentation head
        self.to_fused = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, decoder_dim, 1),
                nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)
            ) for dim in dims
        ])
        self.head = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, 1),
            nn.Conv2d(decoder_dim, num_classes, 1)
        )

        # Domain classifier on pre-MoE features
        self.domain_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # → (B, C, 1, 1)
            nn.Flatten(),              # → (B, C)
            nn.Linear(embed_dims[3], num_domains)  # → (B, D)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        # ... (same init as before) ...
        pass

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    # freeze
    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def forward_features(self, x, pseudo_domain_label):
        B = x.size(0)
        outs = []

        # --- Stage 1 ---
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x, _ = blk(x, H, W, self.expert_num, self.select_mode, pseudo_domain_label, None)
        x = self.norm1(x)  # (B, N1, C1)
        feat1 = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # (B, C1, H, W)
        outs.append(feat1)

        # --- Stage 2 ---
        x, H, W = self.patch_embed2(feat1)
        for blk in self.block2:
            x, _ = blk(x, H, W, self.expert_num, self.select_mode, pseudo_domain_label, None)
        x = self.norm2(x)
        feat2 = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(feat2)

        # --- Stage 3 ---
        x, H, W = self.patch_embed3(feat2)
        for blk in self.block3:
            x, _ = blk(x, H, W, self.expert_num, self.select_mode, pseudo_domain_label, None)
        x = self.norm3(x)
        feat3 = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(feat3)

        # --- Stage 4 pre-MoE (token-level norm) ---
        x, H, W = self.patch_embed4(feat3)  # x: (B, N4, C4)
        x = self.norm4(x)                   # (B, N4, C4)
        pre_moe_feat = x.view(B, H, W, -1)  # (B, C4, H, W)
        pre_moe_feat = pre_moe_feat.permute(0, 3, 1, 2).contiguous()

        # **(여기서 R 계산)**: domain_classifier 마지막 Linear weight로부터
        linear_layer = self.domain_classifier[-1]  # nn.Linear(C4, D)
        W_dom = linear_layer.weight        # (D, C4)
        W_norm = F.normalize(W_dom, dim=1) # (D, C4)
        R = torch.clamp(W_norm @ W_norm.t(), min=0.0)  # (D, D)

        # --- Stage 4 MoE adapters: 각 블록마다 R 인자로 넘겨 MI 계산에 활용 ---
        total_MI_loss = 0.0
        x_moe = x  # (B, N4, C4)
        for blk in self.block4:
            x_moe, mi_loss = blk(x_moe, H, W, self.expert_num, self.select_mode, pseudo_domain_label, R)
            total_MI_loss = total_MI_loss + mi_loss

        x_moe = self.norm4(x_moe)
        feat4 = x_moe.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(feat4)

        return outs, pre_moe_feat, total_MI_loss

    def forward(self, x, pseudo_domain_label):
        layer_outs, pre_moe_feat, total_MI = self.forward_features(x, pseudo_domain_label)

        # segmentation head
        fused = torch.cat([f(o) for f, o in zip(self.to_fused, layer_outs)], dim=1)
        seg_out = self.head(fused)

        # domain prediction
        dom_logits = self.domain_classifier(pre_moe_feat)
        return seg_out, dom_logits, total_MI






class mit_b5(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b5_MOE(MOE_MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b5_MOE, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8],  mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            depths=[3, 6, 40, 3], 
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, 
            expert_num=12, 
            select_mode='new_topk', 
            hidden_dims = [2, 4, 10, 16],        # NOTE you can set them freely 
            num_k = 2
            )   

class mit_b5_MOEv2(MOE_MixVisionTransformerv2):
    def __init__(self, **kwargs):
        super(mit_b5_MOEv2, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8],  mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            depths=[3, 6, 40, 3], 
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, 
            expert_num=12, 
            select_mode='new_topk', 
            hidden_dims = [2, 4, 10, 16],        # NOTE you can set them freely 
            num_k = 2
            )
        
class DomainDiscriminator(nn.Module):
    def __init__(self, n_outputs = 12):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 32 * 32, 250)
        self.fc2 = nn.Linear(250, n_outputs)
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        x = self.pool(F.relu(self.conv1(inputs)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 32 * 32 * 32)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x