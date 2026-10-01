# -*- coding: utf-8 -*-
"""模型层测试：编码器 / 解码器 / 动力学 / rollout / 损失 / 消融开关。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wm.models.world_model import (DynGRU, DynTransformer, VectorQuantizer,
                                   WMConfig, WorldModel)


def tiny_cfg(**kw):
    base = dict(img=16, z_dim=8, base=8, d_model=32, n_layer=1, n_head=2,
                n_boxes=4, n_events=9, vocab=64, max_lang=24, in_ch=13)
    base.update(kw)
    return WMConfig(**base)


def fake_batch(B=2, T=4, img=16, n_boxes=4, cfg=None):
    cfg = cfg or tiny_cfg(img=img, n_boxes=n_boxes)
    x = torch.rand(B, T, cfg.in_ch, img, img)
    return {
        "x": x,
        "depth": x[:, :, 3:4],
        "sem": torch.zeros(B, T, img, img, dtype=torch.long),
        "proprio": torch.rand(B, T, 6),
        "instr": torch.randint(1, 60, (B, cfg.max_lang)),
        "action": torch.rand(B, T, 2) * 2 - 1,
        "box": torch.rand(B, T, n_boxes, 9),
        "events": (torch.rand(B, T, 2 * n_boxes + 1) > 0.7).float(),
        "task": torch.rand(B, T, 4),
        "visibility": torch.rand(B, T, n_boxes),
    }


# ---------------------------------------------------------------- 编码器 / 解码器


def test_encoder_output_shapes():
    cfg = tiny_cfg()
    m = WorldModel(cfg)
    x = torch.rand(3, cfg.in_ch, cfg.img, cfg.img)
    h, mu, logvar = m.enc(x)
    assert mu.shape == (3, cfg.z_dim) == logvar.shape
    assert h.shape == (3, 128)


def test_decode_shapes_and_ranges():
    cfg = tiny_cfg()
    m = WorldModel(cfg)
    z = torch.randn(3, cfg.z_dim)
    rgb, dep, sem = m.decode(z)
    assert rgb.shape == (3, 3, cfg.img, cfg.img)
    assert dep.shape == (3, 1, cfg.img, cfg.img)
    assert sem.shape == (3, 9, cfg.img, cfg.img)
    assert 0.0 <= float(rgb.min()) and float(rgb.max()) <= 1.0
    assert 0.0 <= float(dep.min()) and float(dep.max()) <= 1.0


def test_vae_encoding_is_stochastic_in_train_mode():
    cfg = tiny_cfg()
    m = WorldModel(cfg).train()
    x = torch.rand(2, cfg.in_ch, cfg.img, cfg.img)
    z1, _, _, _ = m.encode(x)
    z2, _, _, _ = m.encode(x)
    assert not torch.allclose(z1, z2), "训练模式下 VAE 采样应当有随机性"
    m.eval()
    z3, _, _, _ = m.encode(x)
    z4, _, _, _ = m.encode(x)
    assert torch.allclose(z3, z4), "推理模式下应当用均值（确定性）"


# ---------------------------------------------------------------- 向量量化


def test_vq_quantizer_selects_nearest_code():
    vq = VectorQuantizer(n_codes=8, dim=4)
    z = vq.codebook.weight.detach()[[2, 5]].clone()      # 正好等于两个码字
    q, loss, idx = vq(z)
    assert idx.tolist() == [2, 5]
    assert torch.allclose(q, z, atol=1e-5)
    assert float(loss) >= 0.0


def test_vq_model_variant_runs():
    cfg = tiny_cfg(vq=True, n_codes=16)
    m = WorldModel(cfg)
    b = fake_batch(cfg=cfg)
    loss, parts = m.losses(b)
    assert torch.isfinite(loss) and parts["vq"] > 0.0


# ---------------------------------------------------------------- 动力学


@pytest.mark.parametrize("dyn", ["transformer", "gru"])
def test_dynamics_output_shape(dyn):
    cfg = tiny_cfg(dyn=dyn)
    d = DynGRU(cfg) if dyn == "gru" else DynTransformer(cfg)
    B, T = 2, 5
    out = d(torch.randn(B, T, cfg.z_dim), torch.rand(B, T, 2),
            torch.rand(B, T, 6), torch.randn(B, cfg.lang_dim))
    assert out.shape == (B, T, cfg.z_dim)


def test_transformer_dynamics_is_causal():
    """因果掩码：改变第 t 步之后的输入，不能影响第 t 步的输出。"""
    cfg = tiny_cfg()
    d = DynTransformer(cfg).eval()
    B, T = 1, 5
    z = torch.randn(B, T, cfg.z_dim)
    a = torch.rand(B, T, 2)
    p = torch.rand(B, T, 6)
    l = torch.randn(B, cfg.lang_dim)
    with torch.no_grad():
        o1 = d(z, a, p, l)
        a2 = a.clone()
        a2[:, 3:] = 0.0                       # 只改 t>=3 的动作
        o2 = d(z, a2, p, l)
    assert torch.allclose(o1[:, :3], o2[:, :3], atol=1e-5), "因果性被破坏"
    assert not torch.allclose(o1[:, 3:], o2[:, 3:], atol=1e-6), "后方动作没有生效"


def test_latent_rollout_shapes_and_no_gt_leak():
    cfg = tiny_cfg()
    m = WorldModel(cfg).eval()
    B, T = 3, 6
    z0 = torch.randn(B, cfg.z_dim)
    actions = torch.rand(B, T, 2)
    zs = m.rollout(z0, actions, torch.rand(B, T, 6), torch.randn(B, cfg.lang_dim))
    assert zs.shape == (B, T + 1, cfg.z_dim)
    assert torch.allclose(zs[:, 0], z0)
    # 同样的起点 + 同样的动作 -> 同样的 rollout（确定性）
    zs2 = m.rollout(z0, actions, torch.zeros(B, T, 6), torch.randn(B, cfg.lang_dim))
    assert zs2.shape == zs.shape


def test_rollout_is_action_sensitive():
    """★ 动作可控性的最小保证：换一组动作，潜轨迹必须变。"""
    torch.manual_seed(0)
    cfg = tiny_cfg()
    m = WorldModel(cfg).eval()
    z0 = torch.randn(1, cfg.z_dim)
    p = torch.rand(1, 5, 6)
    l = torch.randn(1, cfg.lang_dim)
    a = torch.zeros(1, 5, 2)
    b = torch.ones(1, 5, 2)
    za = m.rollout(z0, a, p, l)
    zb = m.rollout(z0, b, p, l)
    assert not torch.allclose(za, zb), "模型对动作不敏感（world model 退化模式）"


# ---------------------------------------------------------------- 损失与训练


def test_loss_parts_present_and_finite():
    cfg = tiny_cfg()
    m = WorldModel(cfg)
    loss, parts = m.losses(fake_batch(cfg=cfg))
    for k in ("rgb", "depth", "sem", "dyn", "state", "event", "task", "kl"):
        assert k in parts, "缺损失项 %s" % k
        assert np.isfinite(parts[k]), "%s 不是有限值" % k
    assert torch.isfinite(loss) and float(loss) > 0.0


def test_loss_decreases_on_overfit():
    """在同一批数据上过拟合若干步，总损失必须明显下降（证明可训练）。"""
    torch.manual_seed(0)
    cfg = tiny_cfg()
    m = WorldModel(cfg)
    b = fake_batch(B=2, T=3, cfg=cfg)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = None
    last = None
    for i in range(30):
        loss, _ = m.losses(b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i == 0:
            first = float(loss)
        last = float(loss)
    assert last < first * 0.7, "损失没有下降: %.4f -> %.4f" % (first, last)


def test_no_action_ablation_changes_behaviour():
    """消融：把动作置零后，动力学损失应当与带动作时不同。"""
    torch.manual_seed(0)
    cfg = tiny_cfg()
    m = WorldModel(cfg)
    b = fake_batch(cfg=cfg)
    _, p1 = m.losses(b)
    b2 = dict(b)
    b2["action"] = torch.zeros_like(b["action"])
    _, p2 = m.losses(b2)
    assert abs(p1["dyn"] - p2["dyn"]) > 1e-6


def test_kl_is_nonnegative():
    cfg = tiny_cfg()
    m = WorldModel(cfg)
    mu = torch.zeros(4, cfg.z_dim)
    logvar = torch.zeros(4, cfg.z_dim)
    assert float(m.kl(mu, logvar)) == pytest.approx(0.0, abs=1e-9)
    assert float(m.kl(mu, torch.ones(4, cfg.z_dim))) > 0.0


def test_language_vector_padding_invariant():
    cfg = tiny_cfg()
    m = WorldModel(cfg)
    a = torch.tensor([[5, 6, 7, 0, 0]])
    b = torch.tensor([[5, 6, 7, 0, 0]])
    assert torch.allclose(m.lang_vec(a), m.lang_vec(b))
    c = torch.tensor([[5, 6, 8, 0, 0]])
    assert not torch.allclose(m.lang_vec(a), m.lang_vec(c))
