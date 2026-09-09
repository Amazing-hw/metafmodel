import torch

from semg_fm.models.ring_criss_cross import RingCrissCrossEncoder


def _encoder() -> RingCrissCrossEncoder:
    return RingCrissCrossEncoder(
        d_model=16,
        depth=2,
        heads=4,
        patch_samples=40,
        channels=16,
        dropout=0.0,
    )


def test_ring_criss_cross_returns_channel_equivariant_and_motion_tokens():
    encoder = _encoder()

    outputs = encoder(torch.randn(2, 400, 16))

    assert outputs["z_eq"].shape == (2, 16, 10, 16)
    assert outputs["z_motion"].shape == (2, 10, 16)


def test_ring_criss_cross_is_circularly_equivariant_in_eval_mode():
    torch.manual_seed(7)
    encoder = _encoder().eval()
    emg = torch.randn(2, 400, 16)

    reference = encoder(emg)
    shifted = encoder(torch.roll(emg, shifts=5, dims=-1))

    torch.testing.assert_close(
        shifted["z_eq"], torch.roll(reference["z_eq"], shifts=5, dims=1), atol=1e-5, rtol=1e-4
    )
    torch.testing.assert_close(shifted["z_motion"], reference["z_motion"], atol=1e-5, rtol=1e-4)


def test_ring_criss_cross_all_missing_mask_has_finite_pooled_outputs():
    encoder = _encoder()
    observed = torch.zeros(2, 16, dtype=torch.bool)

    outputs = encoder(torch.randn(2, 400, 16), electrode_observed=observed)

    assert torch.isfinite(outputs["z_eq"]).all()
    assert torch.isfinite(outputs["z_motion"]).all()
    assert torch.equal(outputs["z_motion"], torch.zeros_like(outputs["z_motion"]))


def test_ring_criss_cross_fusion_gate_is_a_two_way_distribution():
    encoder = _encoder()
    tokens = torch.randn(2, 16, 10, 16)

    gate = encoder.blocks[0].fusion_weights(tokens)

    assert gate.shape == (2, 16, 10, 2)
    torch.testing.assert_close(gate.sum(dim=-1), torch.ones(2, 16, 10))


def test_ring_criss_cross_backpropagates_through_encoder_parameters():
    encoder = _encoder()
    emg = torch.randn(2, 400, 16, requires_grad=True)

    outputs = encoder(emg)
    (outputs["z_eq"].square().mean() + outputs["z_motion"].square().mean()).backward()

    assert emg.grad is not None
    assert torch.isfinite(emg.grad).all()
    assert any(parameter.grad is not None for parameter in encoder.parameters())
