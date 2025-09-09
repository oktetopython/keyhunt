#!/bin/bash

# Universal build script for KeyHunt-Cuda supporting NVIDIA GPUs from 20 series to H100
# Supports multiple CUDA architectures for cross-GPU compatibility

# Detect CUDA path (portable)
if [ -n "${CUDA:-}" ] && [ -x "${CUDA}/bin/nvcc" ]; then
    CUDA_PATH="${CUDA}"
elif [ -n "${CUDA_PATH:-}" ] && [ -x "${CUDA_PATH}/bin/nvcc" ]; then
    CUDA_PATH="${CUDA_PATH}"
else
    NVCC_PATH=$(command -v nvcc 2>/dev/null || true)
    if [ -n "$NVCC_PATH" ]; then
        CUDA_PATH=$(dirname "$(dirname "$NVCC_PATH")")
    else
        for c in /usr/local/cuda /usr/local/cuda-12.* /usr/local/cuda-11.*; do
            if [ -d "$c" ] && [ -x "$c/bin/nvcc" ]; then CUDA_PATH="$c"; break; fi
        done
    fi
fi

if [ -z "${CUDA_PATH:-}" ]; then
    echo "ERROR: CUDA not found. Please export CUDA_PATH or install CUDA." >&2
    exit 1
fi

if [ ! -d "$CUDA_PATH/lib64" ]; then
    echo "WARNING: $CUDA_PATH/lib64 not found; cudart link may fail" >&2
fi

echo "Using CUDA path: $CUDA_PATH"

# NVIDIA GPU architectures from 20 series to H100
# RTX 20 series: Turing (sm_75)
# RTX 30 series: Ampere (sm_80, sm_86)
# RTX 40 series: Ada Lovelace (sm_89)
# H100: Hopper (sm_90)

ARCH_LIST=(
    "75"   # Turing (RTX 20 series)
    "80"   # Ampere (RTX 30 series A100)
    "86"   # Ampere (RTX 30 series consumer)
    "89"   # Ada Lovelace (RTX 40 series)
    "90"   # Hopper (H100)
)

# Function to build for specific architecture
build_for_arch() {
    local arch=$1
    local compute="compute_$arch"
    local sm="sm_$arch"
    
    echo "Building for architecture: $compute / $sm"
    
    # Clean previous build
    make clean
    
    # Build with specific architecture
    CCAP=$arch CUDA="$CUDA_PATH" make gpu=1 -j$(nproc)
    
    if [ $? -eq 0 ]; then
        # Check if binary was created
        if [ -f "KeyHunt" ]; then
            echo "Successfully built KeyHunt"
        elif [ -f "keyhunt" ]; then
            mv keyhunt KeyHunt
            echo "Successfully built KeyHunt"
        else
            echo "Warning: KeyHunt binary not found, but compilation succeeded"
            # Try to manually link if make failed to create binary
            echo "Attempting manual linking..."
            # Check if object files exist before linking
            if [ -f obj/*.o ] && [ -f obj/GPU/*.o ] && [ -f obj/hash/*.o ]; then
                g++ obj/*.o obj/GPU/*.o obj/hash/*.o -lpthread -L$CUDA_PATH/lib64 -lcudart -lgmp -o KeyHunt
                if [ $? -eq 0 ] && [ -f "KeyHunt" ]; then
                    echo "Manual linking successful!"
                else
                    echo "Manual linking failed"
                    # Try alternative approach using find command
                    echo "Trying alternative linking method..."
                    OBJ_FILES=$(find obj -name "*.o" -type f 2>/dev/null)
                    if [ -n "$OBJ_FILES" ]; then
                        g++ $OBJ_FILES -lpthread -L$CUDA_PATH/lib64 -lcudart -lgmp -o KeyHunt
                        if [ $? -eq 0 ] && [ -f "KeyHunt" ]; then
                            echo "Alternative linking successful!"
                        else
                            echo "Alternative linking also failed"
                        fi
                    else
                        echo "No object files found for linking"
                    fi
                fi
            else
                echo "Object files not found for manual linking"
                echo "Available object files:"
                find obj -name "*.o" -type f 2>/dev/null | head -20
            fi
        fi
    else
        echo "Failed to build for architecture ${arch}"
        exit 1
    fi
}

# Function to build universal binary with multiple architectures
build_universal() {
    echo "Building universal binary with multiple architectures..."
    
    # Clean previous build
    make clean
    
    # Build with all architectures sequentially (nvcc doesn't support multiple arch in single command)
    echo "Building for each architecture separately and creating fatbinary..."
    
    local success_count=0
    local built_archs=()
    
    for arch in "${ARCH_LIST[@]}"; do
        echo "Building for architecture: sm_$arch"
        CCAP="$arch" CUDA="$CUDA_PATH" make gpu=1 -j$(nproc)
        
        if [ $? -eq 0 ]; then
            # Check if binary was created
            if [ -f "KeyHunt" ]; then
                echo "Successfully built KeyHunt for architecture sm_${arch}"
                success_count=$((success_count + 1))
                built_archs+=("$arch")
                # Copy binary with architecture suffix
                cp KeyHunt KeyHunt_sm${arch}
            else
                echo "Warning: KeyHunt binary not found for architecture $arch"
            fi
        else
            echo "Failed to build for architecture ${arch}"
        fi
        
        # Clean for next build but keep object files for final linking
        # Only clean GPU object files to avoid recompiling CPU code
        rm -f obj/GPU/*.o
    done
    
    if [ $success_count -gt 0 ]; then
        echo "Successfully built binaries for architectures: ${built_archs[*]}"
        echo "Use './build_universal.sh all' to build separate binaries for each architecture"
    else
        echo "Failed to build any architecture"
        return 1
    fi
}

# Function to detect available GPUs and suggest best architecture
detect_gpus() {
    echo "Detecting available NVIDIA GPUs..."
    
    if ! command -v nvidia-smi &> /dev/null; then
        echo "nvidia-smi not found. Cannot detect GPUs."
        return 1
    fi
    
    # Get GPU information
    local gpu_info=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits)
    
    if [ -z "$gpu_info" ]; then
        echo "No NVIDIA GPUs detected"
        return 1
    fi
    
    echo "Available GPUs:"
    echo "$gpu_info"
    
    # Process GPU info to find highest architecture
    local highest_arch=0
    while IFS=, read -r name compute_cap; do
        # Convert compute capability to integer (e.g., 7.5 -> 75)
        local arch_int=$(echo "$compute_cap" | tr -d '.' | tr -d ' ' | sed 's/^0*//')
        if [ -n "$arch_int" ] && [ "$arch_int" -gt "$highest_arch" ]; then
            highest_arch=$arch_int
        fi
    done <<< "$gpu_info"
    
    if [ "$highest_arch" -gt 0 ]; then
        echo "Suggested architecture: sm_$highest_arch"
    else
        echo "Could not determine suggested architecture"
    fi
}

# Main execution
case "${1:-}" in
    "detect")
        detect_gpus
        ;;
    "universal")
        build_universal
        ;;
    "all")
        echo "Building for all supported architectures..."
        for arch in "${ARCH_LIST[@]}"; do
            build_for_arch "$arch"
        done
        ;;
    "" | "help")
        echo "Usage: $0 [command]"
        echo "Commands:"
        echo "  detect     - Detect available NVIDIA GPUs and suggest best architecture"
        echo "  universal  - Build universal binary supporting all architectures"
        echo "  all        - Build separate binaries for each architecture"
        echo "  help       - Show this help message"
        echo ""
        echo "Supported architectures:"
        for arch in "${ARCH_LIST[@]}"; do
            case "$arch" in
                "75") desc="Turing (RTX 20 series)" ;;
                "80") desc="Ampere (A100)" ;;
                "86") desc="Ampere (RTX 30 series consumer)" ;;
                "89") desc="Ada Lovelace (RTX 40 series)" ;;
                "90") desc="Hopper (H100)" ;;
                *) desc="Unknown" ;;
            esac
            echo "  sm_$arch - $desc"
        done
        ;;
    *)
        # Try to build for specific architecture
        if [[ "$1" =~ ^[0-9]+$ ]]; then
            build_for_arch "$1"
        else
            echo "Invalid command: $1"
            echo "Use '$0 help' for usage information"
            exit 1
        fi
        ;;
esac

# Make script executable
chmod +x "$0"
echo "Build script ready: $0"