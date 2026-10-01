"""动作条件潜在动力学世界模型（action-conditioned latent world model）。

结构（每一块都对应 JD 里明确点名的能力）：

    ┌─ 多模态输入：RGB + 深度 + 语义 one-hot + 本体状态 + 语言指令
    │
    ├─ Encoder（CNN）  -> 潜在 z_t        …… 表示学习
    │     ├ 连续版：VAE（mu/logvar + KL） …… JD: VAE
    │     └ 离散版：VQ 码本（直通估计）    …… JD: VQ-VAE / tokenizer
    │
    ├─ Dynamics（动作条件 Transformer，因果掩码）
    │     token_t = [z_t ; a_t ; proprio_t ; lang]
    │     预测 z_{t+1}                    …… JD: latent dynamics model / autoregressive world model
    │     （另有 GRU 变体用于消融，证明 Transformer 的贡献）
    │
    ├─ Decoder（反卷积） -> RGB / 深度 / 语义重建
    │                                     …… JD: action-conditioned video prediction
    │
    └─ 多任务头（全部从 z 出发）
          ├ 状态头：每个方块 9 维（cx,cy,z,hx,hy,hz,vx,vy,alive）  …… 动力学准确性 / 3D 状态
          ├ 事件头：接触 / 掉桌（2N+1）                          …… 接触事件预测
          └ 任务头：目标距离 / 成功 / 终止 / 存活                 …… 奖励、成功概率、终止状态
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- 配置


@dataclass
class WMConfig:
    in_ch: int = 13              # 3 RGB + 1 depth + 9 semantic
    img: int = 48
    z_dim: int = 32
    vq: bool = False             # True -> 用 VQ 码本（tokenizer 口径）
    n_codes: int = 64
    base: int = 32               # 编码器基础通道
    d_model: int = 128
    n_layer: int = 3
    n_head: int = 4
    dyn: str = "transformer"     # transformer | gru
    n_boxes: int = 4
    n_events: int = 9
    vocab: int = 64
    lang_dim: int = 32
    max_lang: int = 24
    kl_beta: float = 1e-3
    roll_k: int = 4              # 多步潜 rollout 损失的步数（0 = 关掉）
    roll_w: float = 1.0          # 多步 rollout 损失权重
    state_w: float = 2.0         # 状态头权重（"动力学准确性"主要看它）
    recon_w: float = 2.0

    @property
    def down(self) -> int:
        return 8 if self.img % 8 == 0 else 4


# ---------------------------------------------------------------- 编码器


def gn(c: int) -> nn.Module:
    return nn.GroupNorm(min(8, c), c)


class Encoder(nn.Module):
    """48x48 -> 6x6 特征 -> z。步长 2 三次下采样。"""

    def __init__(self, cfg: WMConfig):
        super().__init__()
        b = cfg.base
        self.net = nn.Sequential(
            nn.Conv2d(cfg.in_ch, b, 4, 2, 1), gn(b), nn.SiLU(),
            nn.Conv2d(b, b * 2, 4, 2, 1), gn(b * 2), nn.SiLU(),
            nn.Conv2d(b * 2, b * 3, 4, 2, 1), gn(b * 3), nn.SiLU(),
            nn.Conv2d(b * 3, b * 3, 3, 1, 1), gn(b * 3), nn.SiLU(),
        )
        f = cfg.img // 8
        self.feat = b * 3
        self.f = f
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(b * 3 * f * f, 128), nn.SiLU())
        self.to_mu = nn.Linear(128, cfg.z_dim)
        self.to_logvar = nn.Linear(128, cfg.z_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.head(self.net(x))
        return h, self.to_mu(h), self.to_logvar(h)


class VectorQuantizer(nn.Module):
    """VQ 码本（直通估计）。用于 tokenizer / VQ-VAE 口径的消融。"""

    def __init__(self, n_codes: int, dim: int, beta: float = 0.25):
        super().__init__()
        self.codebook = nn.Embedding(n_codes, dim)
        self.codebook.weight.data.uniform_(-1.0 / n_codes, 1.0 / n_codes)
        self.beta = beta
        self.n_codes = n_codes

    def forward(self, z: torch.Tensor):
        # z: [..., D]
        flat = z.reshape(-1, z.shape[-1])
        d = (flat.pow(2).sum(1, keepdim=True)
             - 2 * flat @ self.codebook.weight.t()
             + self.codebook.weight.pow(2).sum(1)[None, :])
        idx = d.argmin(1)
        q = self.codebook(idx).view_as(z)
        loss = F.mse_loss(q, z.detach()) * self.beta + F.mse_loss(z, q.detach())
        q_st = z + (q - z).detach()          # 直通
        return q_st, loss, idx.view(z.shape[:-1])


class Decoder(nn.Module):
    def __init__(self, cfg: WMConfig):
        super().__init__()
        b = cfg.base
        f = cfg.img // 8
        self.f, self.b, self.feat = f, b, b * 3
        self.proj = nn.Sequential(nn.Linear(cfg.z_dim, b * 3 * f * f), nn.SiLU())
        self.up = nn.Sequential(
            nn.ConvTranspose2d(b * 3, b * 2, 4, 2, 1), gn(b * 2), nn.SiLU(),
            nn.ConvTranspose2d(b * 2, b, 4, 2, 1), gn(b), nn.SiLU(),
            nn.ConvTranspose2d(b, b, 4, 2, 1), gn(b), nn.SiLU(),
        )
        self.to_rgb = nn.Conv2d(b, 3, 3, 1, 1)
        self.to_depth = nn.Conv2d(b, 1, 3, 1, 1)
        self.to_sem = nn.Conv2d(b, cfg.in_ch - 4, 3, 1, 1)

    def forward(self, z: torch.Tensor):
        h = self.proj(z).view(-1, self.feat, self.f, self.f)
        h = self.up(h)
        return (torch.sigmoid(self.to_rgb(h)), torch.sigmoid(self.to_depth(h)),
                self.to_sem(h))


# ---------------------------------------------------------------- 动力学


class DynTransformer(nn.Module):
    """动作条件因果 Transformer：输入 [z_t, a_t, proprio_t, lang]，输出 z_{t+1} 的预测。"""

    def __init__(self, cfg: WMConfig):
        super().__init__()
        d = cfg.d_model
        self.inp = nn.Linear(cfg.z_dim + 2 + 6 + cfg.lang_dim, d)
        layer = nn.TransformerEncoderLayer(d, cfg.n_head, d * 2, dropout=0.0,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.tr = nn.TransformerEncoder(layer, cfg.n_layer)
        self.out = nn.Linear(d, cfg.z_dim)
        self.d = d

    def forward(self, z: torch.Tensor, a: torch.Tensor, p: torch.Tensor,
                lang: torch.Tensor) -> torch.Tensor:
        # z [B,T,Z] a [B,T,2] p [B,T,6] lang [B,Ld]
        B, T, _ = z.shape
        lg = lang[:, None, :].expand(B, T, lang.shape[-1])
        tok = self.inp(torch.cat([z, a, p, lg], dim=-1))
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=z.device), 1)
        h = self.tr(tok, mask=mask)
        return self.out(h)


class DynGRU(nn.Module):
    """GRU 变体，仅用于消融（证明 Transformer 的贡献）。"""

    def __init__(self, cfg: WMConfig):
        super().__init__()
        self.inp = nn.Linear(cfg.z_dim + 2 + 6 + cfg.lang_dim, cfg.d_model)
        self.rnn = nn.GRU(cfg.d_model, cfg.d_model, num_layers=2, batch_first=True)
        self.out = nn.Linear(cfg.d_model, cfg.z_dim)

    def forward(self, z, a, p, lang):
        B, T, _ = z.shape
        lg = lang[:, None, :].expand(B, T, lang.shape[-1])
        h, _ = self.rnn(self.inp(torch.cat([z, a, p, lg], dim=-1)))
        return self.out(h)


# ---------------------------------------------------------------- 完整模型


class WorldModel(nn.Module):
    def __init__(self, cfg: WMConfig):
        super().__init__()
        self.cfg = cfg
        self.enc = Encoder(cfg)
        self.dec = Decoder(cfg)
        self.lang = nn.Embedding(cfg.vocab, cfg.lang_dim, padding_idx=0)
        self.vq = VectorQuantizer(cfg.n_codes, cfg.z_dim) if cfg.vq else None
        self.dyn = DynGRU(cfg) if cfg.dyn == "gru" else DynTransformer(cfg)
        self.state_head = nn.Sequential(
            nn.Linear(cfg.z_dim, 128), nn.SiLU(), nn.Linear(128, cfg.n_boxes * 9))
        self.event_head = nn.Sequential(
            nn.Linear(cfg.z_dim, 128), nn.SiLU(), nn.Linear(128, cfg.n_events))
        self.task_head = nn.Sequential(
            nn.Linear(cfg.z_dim, 64), nn.SiLU(), nn.Linear(64, 4))

    # -------------------------------------------------- 基本操作

    def lang_vec(self, instr: torch.Tensor) -> torch.Tensor:
        e = self.lang(instr)                                  # [B,L,D]
        m = (instr != 0).float()[..., None]
        return (e * m).sum(1) / m.sum(1).clamp(min=1.0)

    def encode(self, x: torch.Tensor, sample: bool = True):
        h, mu, logvar = self.enc(x)
        if self.vq is not None:
            z, vq_loss, idx = self.vq(mu)
            return z, mu, logvar, vq_loss
        if sample and self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        return z, mu, logvar, torch.zeros((), device=x.device)

    def decode(self, z: torch.Tensor):
        return self.dec(z)

    def heads(self, z: torch.Tensor):
        return self.state_head(z), self.event_head(z), self.task_head(z)

    def kl(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def multistep_dyn_loss(self, zv: torch.Tensor, a: torch.Tensor, p: torch.Tensor,
                           lang: torch.Tensor, k: int) -> torch.Tensor:
        """多步潜 rollout 损失（world model 训练的关键一项）。

        为什么必须有它：只做 1 步 teacher forcing 时，动力学网络每步都拿到**真值潜在**，
        推理时却要拿**自己上一步的输出**继续往前推，误差会立刻累积 —— 本项目实测
        1 步训练出的模型 rollout 误差曲线几乎不随步长增长，但绝对值一直很大。
        这里在前 k 步用**预测出来的 z** 继续喂回网络，监督仍对齐编码器给出的真值潜在。
        """
        T = zv.shape[1]
        k = min(k, T - 1)
        if k <= 0:
            return zv.new_zeros(())
        zs = [zv[:, 0]]
        tot = zv.new_zeros(())
        for t in range(k):
            hist = torch.stack(zs, dim=1)                       # [B,t+1,Z]
            pred = self.dyn(hist, a[:, :t + 1], p[:, :t + 1], lang)[:, -1]
            tot = tot + F.mse_loss(pred, zv[:, t + 1].detach())
            zs.append(pred)                                     # 用预测继续
        return tot / k

    # -------------------------------------------------- 潜空间自回归 rollout

    @torch.no_grad()
    def rollout(self, z0: torch.Tensor, actions: torch.Tensor,
                proprio: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        """从 z0 出发，只用**动作序列**在潜空间自回归地预测未来潜在。

        z0        [B,Z]          第 0 帧的真实编码
        actions   [B,T,2]        未来 T 步动作
        proprio   [B,T,6]        本体状态（含步数占比）
        lang      [B,Ld]
        返回       [B,T+1,Z]     含 z0 的潜在轨迹
        """
        self.eval()
        B, T, _ = actions.shape
        zs = [z0]
        for t in range(T):
            z_hist = torch.stack(zs, dim=1)                    # [B,t+1,Z]
            a_in = actions[:, :t + 1]
            p_in = proprio[:, :t + 1]
            pred = self.dyn(z_hist, a_in, p_in, lang)[:, -1]
            zs.append(pred)
        return torch.stack(zs, dim=1)

    # -------------------------------------------------- 损失

    def losses(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.cfg
        x = batch["x"]                                        # [B,T,C,H,W]
        B, T, C, H, W = x.shape
        xf = x.reshape(B * T, C, H, W)
        z, mu, logvar, vq_loss = self.encode(xf)
        rgb, dep, sem = self.decode(z)
        st, ev, tk = self.heads(z)

        loss_rgb = F.l1_loss(rgb, xf[:, :3])
        loss_dep = F.l1_loss(dep, xf[:, 3:4])
        sem_logits = sem.reshape(B * T, sem.shape[1], H * W)
        sem_t = batch["sem"].reshape(B * T, H * W)
        target_ch = torch.zeros_like(sem_t)
        for sid, ch in ((0, 0), (1, 1), (2, 2), (3, 3), (4, 4),
                        (100, 5), (101, 6), (102, 7), (103, 8)):
            target_ch[sem_t == sid] = ch
        loss_sem = F.cross_entropy(sem_logits, target_ch)

        zv = z.reshape(B, T, cfg.z_dim)
        lang = self.lang_vec(batch["instr"])
        pred_next = self.dyn(zv, batch["action"], batch["proprio"], lang)
        loss_dyn = F.mse_loss(pred_next[:, :-1], zv[:, 1:].detach())

        loss_state = F.mse_loss(st.reshape(B, T, -1)[:, :, :cfg.n_boxes * 9],
                                batch["box"].reshape(B, T, -1))
        # 注意：事件/任务头输出是 [B*T, K]，目标在 batch 里是 [B,T,K]，必须先 reshape 对齐
        loss_ev = F.binary_cross_entropy_with_logits(ev, batch["events"].reshape(B * T, -1))
        loss_task = F.mse_loss(tk, batch["task"].reshape(B * T, -1))

        loss_kl = self.kl(mu, logvar)
        loss_roll = self.multistep_dyn_loss(zv, batch["action"], batch["proprio"],
                                            lang, cfg.roll_k)
        total = (cfg.recon_w * loss_rgb + 1.0 * loss_dep + 1.0 * loss_sem
                 + 1.0 * loss_dyn + cfg.roll_w * loss_roll
                 + cfg.state_w * loss_state + 0.5 * loss_ev + 0.5 * loss_task
                 + cfg.kl_beta * loss_kl + vq_loss)
        parts = {"rgb": float(loss_rgb), "depth": float(loss_dep),
                 "sem": float(loss_sem), "dyn": float(loss_dyn),
                 "roll": float(loss_roll),
                 "state": float(loss_state), "event": float(loss_ev),
                 "task": float(loss_task), "kl": float(loss_kl),
                 "vq": float(vq_loss)}
        return total, parts
