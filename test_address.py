#!/usr/bin/env python3

import base58
import hashlib

def address_to_hash160(address):
    """Convert Bitcoin address to hash160"""
    try:
        # Decode base58check
        decoded = base58.b58decode_check(address)
        # Remove version byte (first byte)
        hash160 = decoded[1:]
        return hash160.hex()
    except Exception as e:
        print(f"Error converting address {address}: {e}")
        return None

def test_address():
    address = "17aPYR1m6pVAacXg1PTDDU7XafvK1dxvhi"
    hash160 = address_to_hash160(address)
    print(f"Address: {address}")
    print(f"Hash160: {hash160}")
    
    # Also test the known private key for this address
    # This should be in the range we're searching
    print(f"\nExpected private key should be in range:")
    print(f"Start: 90000000000000")
    print(f"End:   ffffffffffffff")

if __name__ == "__main__":
    test_address()
