#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#include "../../SECP256k1.h"
#include "../../Int.h"

static std::string normalize_hex(const std::string& hex) {
    std::string cleaned;
    cleaned.reserve(hex.size());
    for (char c : hex) {
        if (c != ' ' && c != '\n' && c != '\r' && c != '\t') {
            cleaned.push_back(c);
        }
    }
    if (cleaned.rfind("0x", 0) == 0 || cleaned.rfind("0X", 0) == 0) {
        cleaned.erase(0, 2);
    }
    // Pad to even length for proper parsing
    if (cleaned.size() % 2 != 0) {
        cleaned.insert(cleaned.begin(), '0');
    }
    std::transform(cleaned.begin(), cleaned.end(), cleaned.begin(), ::toupper);
    return cleaned;
}

static std::string pad_scalar(const std::string& hex) {
    if (hex.size() >= 64) {
        return hex;
    }
    std::string padded = std::string(64 - hex.size(), '0');
    padded += hex;
    return padded;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: keyhunt_secp_bridge <hex_scalar> [<hex_scalar> ...]" << std::endl;
        return 1;
    }

    std::vector<std::string> scalars;
    scalars.reserve(static_cast<size_t>(argc) - 1);
    for (int i = 1; i < argc; ++i) {
        scalars.push_back(pad_scalar(normalize_hex(argv[i])));
    }

    try {
        Secp256K1 secp;
        secp.SetFastInit(true);   // use fast init for runtime-sensitive baseline generation
        secp.Init();

        for (const std::string& scalarHex : scalars) {
            Int k;
            k.SetBase16(scalarHex.c_str());

            Point pub = secp.ComputePublicKey(&k);

            std::string xHex = pub.x.GetBase16();
            std::string yHex = pub.y.GetBase16();

            // Results prefixed with RESULT for easy parsing
            std::cout << "RESULT " << scalarHex << ' ' << xHex << ' ' << yHex << std::endl;
        }
    } catch (const std::exception& ex) {
        std::cerr << "ERROR " << ex.what() << std::endl;
        return 2;
    }

    return 0;
}
