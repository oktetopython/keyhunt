import ctypes
import os
import random
import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLE_HEADER = PROJECT_ROOT / "PrecomputedTables.h"


def _clean_value(token: str) -> int:
    token = token.strip()
    token = token.removesuffix("ULL").removesuffix("ull").removesuffix("UL").removesuffix("ul")
    return int(token, 16)


def _parse_array(text: str, name: str) -> list[int]:
    pattern = re.compile(rf"{name}\s*\[4\]\s*=\s*\{{([^}}]+)\}}", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Array {name} not found in PrecomputedTables.h")
    values = [_clean_value(token) for token in match.group(1).split(',') if token.strip()]
    if len(values) != 4:
        raise ValueError(f"Array {name} expected 4 limbs, found {len(values)}")
    return values


def _limbs_to_int(limbs: list[int]) -> int:
    value = 0
    for idx, limb in enumerate(limbs):
        value |= limb << (64 * idx)
    return value


def _limbs_to_hex(limbs: list[int]) -> str:
    return f"{_limbs_to_int(limbs):064X}"


def _int_to_bytes(value: int) -> bytes:
    return value.to_bytes(32, byteorder="big")


def parse_precomputed_tables():
    text = TABLE_HEADER.read_text()
    data = {
        "field_p": _limbs_to_int(_parse_array(text, "FIELD_P")),
        "order_n": _limbs_to_int(_parse_array(text, "ORDER_N")),
        "generator": (
            _limbs_to_hex(_parse_array(text, "GENERATOR_X")),
            _limbs_to_hex(_parse_array(text, "GENERATOR_Y")),
        ),
        "lambda": _limbs_to_int(_parse_array(text, "LAMBDA")),
        "beta": _limbs_to_int(_parse_array(text, "BETA")),
        "multiples": {},
    }

    for n in range(1, 17):
        x = _limbs_to_hex(_parse_array(text, f"MULT_{n}G_X"))
        y = _limbs_to_hex(_parse_array(text, f"MULT_{n}G_Y"))
        data["multiples"][n] = (x, y)

    return data


class Libsecp256k1Wrapper:
    """Thin ctypes wrapper for libsecp256k1"""

    def __init__(self):
        self._lib = ctypes.CDLL("libsecp256k1.so")
        self._configure()
        flags = (1 << 0) | (1 << 9) | (1 << 8)  # CONTEXT_SIGN | CONTEXT_VERIFY
        self._ctx = self._lib.secp256k1_context_create(ctypes.c_uint(flags))
        if not self._ctx:
            raise RuntimeError("Failed to create secp256k1 context")

    def __del__(self):
        ctx = getattr(self, "_ctx", None)
        if ctx:
            self._lib.secp256k1_context_destroy(ctx)
            self._ctx = None

    def _configure(self):
        self._lib.secp256k1_context_create.argtypes = [ctypes.c_uint]
        self._lib.secp256k1_context_create.restype = ctypes.c_void_p
        self._lib.secp256k1_context_destroy.argtypes = [ctypes.c_void_p]
        self._lib.secp256k1_context_destroy.restype = None

        class PubKey(ctypes.Structure):
            _fields_ = [("data", ctypes.c_ubyte * 64)]

        self._pubkey_type = PubKey

        self._lib.secp256k1_ec_pubkey_create.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(PubKey),
            ctypes.c_char_p,
        ]
        self._lib.secp256k1_ec_pubkey_create.restype = ctypes.c_int

        self._lib.secp256k1_ec_pubkey_serialize.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(PubKey),
            ctypes.c_uint,
        ]
        self._lib.secp256k1_ec_pubkey_serialize.restype = ctypes.c_int

    def compute_point(self, scalar_bytes: bytes) -> tuple[str, str]:
        if len(scalar_bytes) != 32:
            raise ValueError("Scalar must be 32 bytes")

        seckey = ctypes.create_string_buffer(scalar_bytes)
        pubkey = self._pubkey_type()

        res = self._lib.secp256k1_ec_pubkey_create(self._ctx, ctypes.byref(pubkey), seckey)
        if res != 1:
            raise ValueError("Invalid private key for secp256k1")

        output_len = ctypes.c_size_t(65)
        output = ctypes.create_string_buffer(output_len.value)
        flags_uncompressed = 1 << 1  # SECP256K1_EC_UNCOMPRESSED
        res = self._lib.secp256k1_ec_pubkey_serialize(
            self._ctx,
            output,
            ctypes.byref(output_len),
            ctypes.byref(pubkey),
            ctypes.c_uint(flags_uncompressed),
        )
        if res != 1:
            raise RuntimeError("Failed to serialize secp256k1 public key")

        data = output.raw[: output_len.value]
        if len(data) != 65 or data[0] != 0x04:
            raise RuntimeError("Unexpected secp256k1 public key format")

        x = data[1:33]
        y = data[33:65]
        return x.hex().upper(), y.hex().upper()


class LibsecpBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.constants = parse_precomputed_tables()
        cls.lib = Libsecp256k1Wrapper()

    def test_generator_matches_libsecp(self):
        scalar_one = _int_to_bytes(1)
        x_ref, y_ref = self.lib.compute_point(scalar_one)
        x_tbl, y_tbl = self.constants["generator"]
        self.assertEqual(x_ref, x_tbl)
        self.assertEqual(y_ref, y_tbl)

    def test_precomputed_multiples(self):
        for n, (x_tbl, y_tbl) in self.constants["multiples"].items():
            scalar_bytes = _int_to_bytes(n)
            x_ref, y_ref = self.lib.compute_point(scalar_bytes)
            self.assertEqual(x_ref, x_tbl, f"X mismatch for {n}*G")
            self.assertEqual(y_ref, y_tbl, f"Y mismatch for {n}*G")

    def test_glv_endomorphism_constants(self):
        lam_bytes = _int_to_bytes(self.constants["lambda"] % self.constants["order_n"])
        x_lambda, y_lambda = self.lib.compute_point(lam_bytes)

        gen_x_hex, gen_y_hex = self.constants["generator"]
        gen_x = int(gen_x_hex, 16)
        beta = self.constants["beta"]
        field_p = self.constants["field_p"]

        x_phi = (beta * gen_x) % field_p
        self.assertEqual(x_lambda, f"{x_phi:064X}")
        self.assertEqual(y_lambda, gen_y_hex)

    def test_glv_fuzz_against_libsecp(self):
        if not os.environ.get("KEYHUNT_ENABLE_FUZZ"):
            self.skipTest("Fuzzing disabled. Set KEYHUNT_ENABLE_FUZZ=1 to enable.")

        rng = random.Random(0xB1A55)
        order = self.constants["order_n"]
        beta = self.constants["beta"]
        field_p = self.constants["field_p"]
        lam = self.constants["lambda"] % order

        for _ in range(128):
            k = rng.randrange(1, order)
            x_k, y_k = self.lib.compute_point(_int_to_bytes(k))

            x_int = int(x_k, 16)
            phi_x = (beta * x_int) % field_p

            lam_k = (lam * k) % order
            x_lam_k, y_lam_k = self.lib.compute_point(_int_to_bytes(lam_k))

            self.assertEqual(x_lam_k, f"{phi_x:064X}")
            self.assertEqual(y_lam_k, y_k)


if __name__ == "__main__":
    unittest.main()
