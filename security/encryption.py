"""Pluggable encryption backends for federated updates."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class EncryptionBackend:
    """Interface for encryption backends."""

    name: str = "base"

    def setup(self, num_clients: int, vector_dim: int) -> None:
        raise NotImplementedError

    def encrypt_update(self, update: np.ndarray, client_id: int) -> Any:
        raise NotImplementedError

    def decrypt_update(self, ciphertext: Any) -> np.ndarray:
        raise NotImplementedError

    def aggregate(self, ciphertexts: list[Any]) -> Any:
        raise NotImplementedError

    def decrypt_aggregate(self, aggregate_ciphertext: Any, participant_count: int) -> np.ndarray:
        raise NotImplementedError


@dataclass
class SimulatedCiphertext:
    """Toy ciphertext for simulation mode (not cryptographically secure)."""

    masked_values: np.ndarray
    mask: np.ndarray


class SimulatedEncryptionBackend(EncryptionBackend):
    """Noise-mask encryption simulation for local development and demos."""

    name = "simulated"

    def __init__(self, mask_std: float = 0.002, seed: int = 42) -> None:
        self.mask_std = mask_std
        self.rng = np.random.default_rng(seed)
        self.vector_dim = 0

    def setup(self, num_clients: int, vector_dim: int) -> None:
        _ = num_clients
        self.vector_dim = vector_dim

    def encrypt_update(self, update: np.ndarray, client_id: int) -> SimulatedCiphertext:
        _ = client_id
        mask = self.rng.normal(0.0, self.mask_std, size=update.shape)
        return SimulatedCiphertext(masked_values=update + mask, mask=mask)

    def decrypt_update(self, ciphertext: SimulatedCiphertext) -> np.ndarray:
        return ciphertext.masked_values - ciphertext.mask

    def aggregate(self, ciphertexts: list[SimulatedCiphertext]) -> SimulatedCiphertext:
        if not ciphertexts:
            raise ValueError("No ciphertexts to aggregate")
        masked_sum = np.sum([c.masked_values for c in ciphertexts], axis=0)
        mask_sum = np.sum([c.mask for c in ciphertexts], axis=0)
        return SimulatedCiphertext(masked_values=masked_sum, mask=mask_sum)

    def decrypt_aggregate(
        self,
        aggregate_ciphertext: SimulatedCiphertext,
        participant_count: int,
    ) -> np.ndarray:
        if participant_count <= 0:
            raise ValueError("participant_count must be positive")
        summed = self.decrypt_update(aggregate_ciphertext)
        return summed / participant_count


class OpenFHEEncryptionBackend(EncryptionBackend):
    """
    Wrapper around existing OpenFHE binaries via `openfhe_lib/<scheme>/openFHE.py`.

    Notes:
    - Current C++ aggregator is fixed to 4 client ciphertext files.
    - The Python wrapper divides decrypted values by 4 internally.
    - Vector slot length in client decrypt is fixed to 32.
    """

    name = "openfhe"

    def __init__(self, scheme: str = "bgv") -> None:
        self.scheme = scheme.lower()
        if self.scheme not in {"bgv", "bfv"}:
            raise ValueError("OpenFHE backend supports scheme='bgv' or 'bfv'")

        self._module = None
        self.vector_dim = 0
        self.slot_dim = 32
        self.fixed_client_count = 4

    def setup(self, num_clients: int, vector_dim: int) -> None:
        if num_clients != self.fixed_client_count:
            raise ValueError(
                "OpenFHE backend currently requires num_clients=4 "
                "(matches existing C++ server/client wrapper)."
            )
        if vector_dim > self.slot_dim:
            raise ValueError(
                f"Model dimension {vector_dim} exceeds OpenFHE wrapper slot size {self.slot_dim}."
            )

        self.vector_dim = vector_dim
        self._module = importlib.import_module(f"openfhe_lib.{self.scheme}.openFHE")

    def _pad(self, update: np.ndarray) -> np.ndarray:
        if update.shape[0] > self.slot_dim:
            raise ValueError("Update larger than OpenFHE slot size")
        if update.shape[0] == self.slot_dim:
            return update
        padded = np.zeros(self.slot_dim, dtype=np.float64)
        padded[: update.shape[0]] = update
        return padded

    def _trim(self, values: np.ndarray) -> np.ndarray:
        return values[: self.vector_dim]

    def _client_cipher_file(self, client_id: int) -> str:
        return f"/enc_weight_client{client_id + 1}.txt"

    def _aggregate_file(self) -> str:
        return "/enc_aggregator_weight_server.txt"

    def encrypt_update(self, update: np.ndarray, client_id: int) -> str:
        if self._module is None:
            raise RuntimeError("OpenFHE backend not initialized. Call setup() first.")
        padded = self._pad(update)
        cipher_file = self._client_cipher_file(client_id)
        self._module.encrypt_weights(padded.tolist(), cipher_file)
        return cipher_file

    def decrypt_update(self, ciphertext: str) -> np.ndarray:
        if self._module is None:
            raise RuntimeError("OpenFHE backend not initialized. Call setup() first.")
        # Existing wrapper divides by 4 during decrypt; invert for single-client recovery.
        values = np.array(self._module.decrypt_weights(ciphertext), dtype=np.float64) * 4.0
        return self._trim(values)

    def aggregate(self, ciphertexts: list[str]) -> str:
        _ = ciphertexts
        if self._module is None:
            raise RuntimeError("OpenFHE backend not initialized. Call setup() first.")
        # Server always reads fixed files: enc_weight_client1..4.txt
        self._module.aggregator()
        return self._aggregate_file()

    def decrypt_aggregate(self, aggregate_ciphertext: str, participant_count: int) -> np.ndarray:
        if self._module is None:
            raise RuntimeError("OpenFHE backend not initialized. Call setup() first.")
        if participant_count <= 0:
            raise ValueError("participant_count must be positive")
        # Wrapper returns (sum / 4). Convert to mean over active participants.
        values = np.array(self._module.decrypt_weights(aggregate_ciphertext), dtype=np.float64)
        scaled = values * (self.fixed_client_count / float(participant_count))
        return self._trim(scaled)

    def overwrite_client_with_zero(self, client_id: int) -> None:
        """Re-encrypt a zero update for dropped clients before aggregation."""
        zero_update = np.zeros(self.vector_dim, dtype=np.float64)
        self.encrypt_update(zero_update, client_id)

    def ensure_required_paths(self) -> None:
        """Ensure OpenFHE data path exists for scheme."""
        scheme_path = Path("data") / self.scheme
        scheme_path.mkdir(parents=True, exist_ok=True)


def build_encryption_backend(
    backend_type: str,
    scheme: str,
    seed: int,
) -> EncryptionBackend:
    """Factory for encryption backend instances."""
    backend_type = backend_type.lower()
    if backend_type == "simulated":
        return SimulatedEncryptionBackend(seed=seed)
    if backend_type == "openfhe":
        return OpenFHEEncryptionBackend(scheme=scheme)
    raise ValueError(f"Unsupported encryption backend: {backend_type}")
