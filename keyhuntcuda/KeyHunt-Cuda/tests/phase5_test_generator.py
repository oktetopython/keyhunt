#!/usr/bin/env python3
"""
Phase 5 End-to-End Test Data Generator
Generates test cases for PUZZLE71 system integration testing
"""

import os
import sys
import random
import hashlib
import json
import argparse
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import numpy as np

# SECP256K1 parameters
SECP256K1_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
PUZZLE71_START = 0x800000000000000000  # 2^70
PUZZLE71_END = 0xFFFFFFFFFFFFFFFFFF   # 2^71 - 1
PUZZLE71_TARGET = "3EE4133D991F52FDF6A25C9834E0745AC74248A4"  # Known target address

@dataclass
class TestCase:
    """Represents a single test case for PUZZLE71 testing"""
    test_id: str
    private_key: int
    public_key_x: str
    public_key_y: str
    compressed_pub: str
    bitcoin_address: str
    search_range_start: int
    search_range_end: int
    expected_found: bool
    description: str

class TestDataGenerator:
    """Generates test data for Phase 5 system integration testing"""
    
    def __init__(self, seed: int = None):
        """Initialize generator with optional seed for reproducibility"""
        if seed:
            random.seed(seed)
            np.random.seed(seed)
        
    def generate_private_key(self, min_val: int = None, max_val: int = None) -> int:
        """Generate a random private key within specified range"""
        if min_val is None:
            min_val = 1
        if max_val is None:
            max_val = SECP256K1_N - 1
        
        return random.randint(min_val, max_val)
    
    def private_to_public(self, private_key: int) -> Tuple[int, int]:
        """
        Simplified EC point multiplication (mock implementation)
        In production, this would use actual secp256k1 operations
        """
        # Mock implementation - returns deterministic but fake values
        # Real implementation would use secp256k1 library
        x = (private_key * 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798) % SECP256K1_N
        y = (private_key * 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8) % SECP256K1_N
        return (x, y)
    
    def public_to_address(self, x: int, y: int, compressed: bool = True) -> str:
        """
        Convert public key to Bitcoin address (mock implementation)
        In production, this would use actual Bitcoin address generation
        """
        # Mock implementation - returns deterministic but fake address
        if compressed:
            prefix = b'\x02' if y % 2 == 0 else b'\x03'
            pub_bytes = prefix + x.to_bytes(32, 'big')
        else:
            pub_bytes = b'\x04' + x.to_bytes(32, 'big') + y.to_bytes(32, 'big')
        
        # Simplified hash (real would use SHA256 + RIPEMD160)
        hash_val = hashlib.sha256(pub_bytes).hexdigest()[:40]
        return hash_val.upper()
    
    def generate_test_case(self, 
                          test_type: str,
                          index: int,
                          target_key: int = None) -> TestCase:
        """Generate a single test case based on type"""
        
        test_id = f"{test_type}_{index:04d}"
        
        if test_type == "positive":
            # Generate test with known key in range
            if target_key:
                private_key = target_key
            else:
                private_key = self.generate_private_key(PUZZLE71_START, PUZZLE71_END)
            
            # Small search range around the key
            offset = random.randint(1000, 1000000)
            search_start = max(PUZZLE71_START, private_key - offset)
            search_end = min(PUZZLE71_END, private_key + offset)
            expected_found = True
            description = f"Positive test: key {hex(private_key)} in range"
            
        elif test_type == "negative":
            # Generate test with key outside search range
            private_key = self.generate_private_key(1, PUZZLE71_START - 1)
            search_start = PUZZLE71_START
            search_end = PUZZLE71_START + random.randint(1000000, 10000000)
            expected_found = False
            description = f"Negative test: key {hex(private_key)} outside range"
            
        elif test_type == "boundary":
            # Test boundary conditions
            if index % 2 == 0:
                # Test at start boundary
                private_key = PUZZLE71_START
                search_start = PUZZLE71_START
                search_end = PUZZLE71_START + 1000000
            else:
                # Test at end boundary
                private_key = PUZZLE71_END
                search_start = PUZZLE71_END - 1000000
                search_end = PUZZLE71_END
            expected_found = True
            description = f"Boundary test: key at {'start' if index % 2 == 0 else 'end'} of range"
            
        elif test_type == "large_range":
            # Test with large search range
            private_key = self.generate_private_key(PUZZLE71_START, PUZZLE71_END)
            # Large range (1/1000th of full range)
            range_size = (PUZZLE71_END - PUZZLE71_START) // 1000
            search_start = max(PUZZLE71_START, private_key - range_size // 2)
            search_end = min(PUZZLE71_END, search_start + range_size)
            expected_found = search_start <= private_key <= search_end
            description = f"Large range test: {hex(range_size)} keys to search"
            
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        # Generate public key and address
        x, y = self.private_to_public(private_key)
        pub_x = hex(x)[2:].zfill(64)
        pub_y = hex(y)[2:].zfill(64)
        compressed = ('02' if y % 2 == 0 else '03') + pub_x
        address = self.public_to_address(x, y)
        
        return TestCase(
            test_id=test_id,
            private_key=private_key,
            public_key_x=pub_x,
            public_key_y=pub_y,
            compressed_pub=compressed,
            bitcoin_address=address,
            search_range_start=search_start,
            search_range_end=search_end,
            expected_found=expected_found,
            description=description
        )
    
    def generate_test_suite(self, suite_name: str = "default") -> Dict:
        """Generate complete test suite for Phase 5 testing"""
        
        test_cases = []
        
        # Test suite configuration
        configs = {
            "small": {
                "positive": 5,
                "negative": 3,
                "boundary": 2,
                "large_range": 2
            },
            "medium": {
                "positive": 20,
                "negative": 10,
                "boundary": 4,
                "large_range": 6
            },
            "large": {
                "positive": 100,
                "negative": 50,
                "boundary": 10,
                "large_range": 20
            },
            "default": {
                "positive": 10,
                "negative": 5,
                "boundary": 4,
                "large_range": 4
            }
        }
        
        config = configs.get(suite_name, configs["default"])
        
        # Generate test cases for each type
        for test_type, count in config.items():
            for i in range(count):
                test_case = self.generate_test_case(test_type, i)
                test_cases.append(test_case)
        
        # Add special test case for known PUZZLE71 target
        if suite_name in ["large", "default"]:
            # This would be the actual PUZZLE71 key if known
            # For testing, we use a mock value in the range
            special_case = TestCase(
                test_id="special_puzzle71",
                private_key=PUZZLE71_START + 0x123456789ABCDEF,  # Mock value
                public_key_x="0" * 64,  # Would be actual public key
                public_key_y="0" * 64,
                compressed_pub="02" + "0" * 64,
                bitcoin_address=PUZZLE71_TARGET,
                search_range_start=PUZZLE71_START,
                search_range_end=PUZZLE71_END,
                expected_found=True,
                description="Special test: PUZZLE71 target address"
            )
            test_cases.append(special_case)
        
        # Create test suite metadata
        suite = {
            "suite_name": suite_name,
            "total_tests": len(test_cases),
            "puzzle71_range": {
                "start": hex(PUZZLE71_START),
                "end": hex(PUZZLE71_END),
                "size": PUZZLE71_END - PUZZLE71_START + 1
            },
            "test_distribution": {
                "positive_tests": len([t for t in test_cases if t.expected_found]),
                "negative_tests": len([t for t in test_cases if not t.expected_found])
            },
            "test_cases": [asdict(tc) for tc in test_cases]
        }
        
        return suite
    
    def save_test_suite(self, suite: Dict, filename: str):
        """Save test suite to JSON file"""
        # Convert integer values to hex strings for readability
        def convert_to_hex(obj):
            if isinstance(obj, dict):
                return {k: convert_to_hex(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_hex(item) for item in obj]
            elif isinstance(obj, int) and obj > 1000000:
                return hex(obj)
            return obj
        
        suite_hex = convert_to_hex(suite)
        
        with open(filename, 'w') as f:
            json.dump(suite_hex, f, indent=2)
        
        print(f"Test suite saved to {filename}")
    
    def generate_batch_config(self, num_batches: int = 10) -> List[Dict]:
        """Generate batch processing configuration for testing"""
        
        batch_configs = []
        total_range = PUZZLE71_END - PUZZLE71_START + 1
        batch_size = total_range // num_batches
        
        for i in range(num_batches):
            start = PUZZLE71_START + (i * batch_size)
            end = start + batch_size - 1 if i < num_batches - 1 else PUZZLE71_END
            
            config = {
                "batch_id": f"batch_{i:03d}",
                "range_start": hex(start),
                "range_end": hex(end),
                "batch_size": end - start + 1,
                "gpu_assignment": i % 4,  # Assign to GPU 0-3 cyclically
                "priority": random.randint(1, 10),
                "checkpoint_enabled": True,
                "checkpoint_interval": 1000000
            }
            batch_configs.append(config)
        
        return batch_configs

def main():
    """Main entry point for test data generation"""
    
    parser = argparse.ArgumentParser(description="Generate test data for Phase 5 testing")
    parser.add_argument("--suite", default="default", 
                       choices=["small", "medium", "large", "default"],
                       help="Test suite size")
    parser.add_argument("--output", default="phase5_test_data.json",
                       help="Output filename for test data")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducible tests")
    parser.add_argument("--batches", type=int, default=10,
                       help="Number of batch configurations to generate")
    
    args = parser.parse_args()
    
    # Create generator
    generator = TestDataGenerator(seed=args.seed)
    
    # Generate test suite
    print(f"Generating {args.suite} test suite...")
    test_suite = generator.generate_test_suite(args.suite)
    
    # Generate batch configurations
    print(f"Generating {args.batches} batch configurations...")
    batch_configs = generator.generate_batch_config(args.batches)
    test_suite["batch_configs"] = batch_configs
    
    # Save to file
    generator.save_test_suite(test_suite, args.output)
    
    # Print summary
    print("\nTest Suite Summary:")
    print(f"  Total test cases: {test_suite['total_tests']}")
    print(f"  Positive tests: {test_suite['test_distribution']['positive_tests']}")
    print(f"  Negative tests: {test_suite['test_distribution']['negative_tests']}")
    print(f"  Batch configurations: {len(batch_configs)}")
    print(f"\nPUZZLE71 Range:")
    print(f"  Start: {test_suite['puzzle71_range']['start']}")
    print(f"  End: {test_suite['puzzle71_range']['end']}")
    print(f"  Size: {test_suite['puzzle71_range']['size']:,}")

if __name__ == "__main__":
    main()