# src/encryption/context.py

import tenseal as ts
from pathlib import Path


class EncryptionContextManager:
    """
    Creates, saves, and loads a CKKS context tuned for encrypted neural networks.
    """

    def __init__(self, security_level="128bit"):
        """
        Upgraded parameters:
            poly_modulus_degree: 16384
            coeff_mod_bit_sizes: [60, 45, 45, 45, 60]
            global_scale: 2^30
        """

        if security_level == "128bit":
            self.poly_modulus_degree = 16384
            self.coeff_mod_bit_sizes = [60, 45, 45, 45, 60]
        else:
            raise ValueError("Unsupported security level")

        # Lower starting scale → safer NN inference
        self.global_scale = 2 ** 30
        self._ctx = None

    # ----------------------------------------------------------
    # Create a new CKKS context
    # ----------------------------------------------------------
    def create_context(self):
        ctx = ts.context(
            scheme=ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.poly_modulus_degree,
            coeff_mod_bit_sizes=self.coeff_mod_bit_sizes,
        )

        ctx.global_scale = self.global_scale
        ctx.generate_galois_keys()
        ctx.generate_relin_keys()
        ctx.auto_relin = True

        self._ctx = ctx
        return ctx

    # ----------------------------------------------------------
    # Save context (without secret key)
    # ----------------------------------------------------------
    def save_context(self, path: str):
        if self._ctx is None:
            raise RuntimeError("No context initialized.")

        ctx_bytes = self._ctx.serialize(save_secret_key=False)
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            f.write(ctx_bytes)

    # ----------------------------------------------------------
    # Load context
    # ----------------------------------------------------------
    def load_context(self, path: str):
        with open(path, "rb") as f:
            data = f.read()

        ctx = ts.context_from(data)
        ctx.global_scale = self.global_scale
        ctx.auto_relin = True

        self._ctx = ctx
        return ctx

    # ----------------------------------------------------------
    # Public context info
    # ----------------------------------------------------------
    def get_context_info(self):
        return {
            "poly_modulus_degree": self.poly_modulus_degree,
            "coeff_mod_bit_sizes": self.coeff_mod_bit_sizes,
        }