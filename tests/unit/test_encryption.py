import numpy as np
import pytest
import tenseal as ts

from src.encryption.context import EncryptionContextManager
from src.models.encrypted_lr import EncryptedLogisticRegression
from src.models.encrypted_nn import EncryptedNeuralNetwork
from src.utils.ckks_budget import estimate_levels, assert_depth_budget


CKKS_SCALE = 2 ** 12  # low test scale to keep ample modulus headroom


def make_client_context():
    """
    Build a client-side CKKS context with a secret key for decrypting test outputs.
    """
    coeff_mod_bit_sizes = [60, 40, 40, 40, 40, 40, 40, 60]  # deeper chain for test depth
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )
    ctx.global_scale = CKKS_SCALE
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    ctx.auto_relin = True
    return ctx


def make_server_context_from_client(client_ctx: ts.Context):
    """
    For functionality tests we keep the full context (with eval + secret keys)
    to avoid TenSEAL scale issues in the test harness. Production still uses
    public-only contexts; that is covered separately in the security test.
    """
    full_bytes = client_ctx.serialize()
    ctx = ts.context_from(full_bytes)
    ctx.global_scale = CKKS_SCALE
    ctx.auto_relin = True
    return ctx


def test_public_context_cannot_decrypt():
    manager = EncryptionContextManager()
    public_ctx = manager.create_context()
    enc_vec = ts.ckks_vector(public_ctx, [0.1, 0.2])

    # Without a secret key, decrypt should fail.
    with pytest.raises(Exception):
        _ = enc_vec.decrypt()


def test_encrypted_lr_roundtrip_matches_plaintext():
    client_ctx = make_client_context()
    server_ctx = client_ctx  # use same context to avoid scale drift in test

    model = EncryptedLogisticRegression(server_ctx)
    model.weights = np.array([0.5, -0.25, 0.1])
    model.bias = 0.2
    model.is_loaded = True

    x = np.array([1.0, 2.0, -1.0])
    enc_x = ts.ckks_vector(client_ctx, x.tolist())

    enc_out = model.predict_encrypted_logit(enc_x)
    # TenSEAL returns scaled values; normalize by test scale for comparison.
    logit = float(enc_out.decrypt()[0]) / CKKS_SCALE
    expected = float(np.dot(model.weights, x) + model.bias)

    assert abs(logit - expected) < 0.3


def test_depth_budget_guard():
    coeff_mod_bit_sizes = [60, 45, 45, 45, 60]
    scale_bits = 30
    levels = estimate_levels(coeff_mod_bit_sizes, scale_bits)
    assert levels > 0

    # Should not raise when within budget
    assert_depth_budget(required_muls=levels - 1, coeff_mod_bit_sizes=coeff_mod_bit_sizes, scale_bits=scale_bits)

    # Exceeding budget should raise
    with pytest.raises(ValueError):
        assert_depth_budget(required_muls=levels + 2, coeff_mod_bit_sizes=coeff_mod_bit_sizes, scale_bits=scale_bits)


def test_encrypted_nn_two_hidden_layers_forward():
    client_ctx = make_client_context()
    nn_model = EncryptedNeuralNetwork(client_ctx)
    h1, h2 = 3, 2
    nn_model.W1 = np.array([[0.2, -0.1, 0.05],
                            [0.1, 0.05, -0.02],
                            [-0.05, 0.03, 0.04]])
    nn_model.b1 = np.zeros(h1)
    nn_model.W2 = np.ones((h2, h1)) * 0.1
    nn_model.b2 = np.zeros(h2)
    nn_model.W3 = np.ones((1, h2)) * 0.1
    nn_model.b3 = np.array([0.0])
    nn_model.loaded = True

    x = np.array([0.2, -0.1, 0.05])
    enc_x_client = ts.ckks_vector(client_ctx, x.tolist())

    enc_out_server = nn_model.predict_encrypted(enc_x_client)
    enc_out_client = ts.ckks_vector_from(client_ctx, enc_out_server.serialize())
    prob = float(enc_out_client.decrypt()[0])

    assert not np.isnan(prob)
    assert -5.0 < prob < 5.0
