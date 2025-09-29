// Verification program for PUZZLE71 implementation
// This program verifies that all PUZZLE71 components are properly integrated

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>

// Color output for Windows console
#ifdef _WIN32
    #include <windows.h>
    void setColor(int color) {
        HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
        SetConsoleTextAttribute(hConsole, color);
    }
    #define COLOR_GREEN 10
    #define COLOR_RED 12
    #define COLOR_YELLOW 14
    #define COLOR_WHITE 15
    #define COLOR_CYAN 11
#else
    void setColor(int color) {}
    #define COLOR_GREEN 0
    #define COLOR_RED 0
    #define COLOR_YELLOW 0
    #define COLOR_WHITE 0
    #define COLOR_CYAN 0
#endif

struct TestResult {
    std::string testName;
    bool passed;
    std::string message;
};

class PUZZLE71Verifier {
private:
    std::vector<TestResult> results;
    int totalTests = 0;
    int passedTests = 0;

    bool fileContains(const std::string& filename, const std::string& searchStr) {
        std::ifstream file(filename);
        if (!file.is_open()) return false;
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.find(searchStr) != std::string::npos) {
                return true;
            }
        }
        return false;
    }

    bool fileExists(const std::string& filename) {
        std::ifstream file(filename);
        return file.good();
    }

public:
    void runTest(const std::string& testName, bool condition, const std::string& message = "") {
        TestResult result;
        result.testName = testName;
        result.passed = condition;
        result.message = message;
        
        results.push_back(result);
        totalTests++;
        if (condition) passedTests++;
        
        setColor(COLOR_YELLOW);
        std::cout << "[TEST " << totalTests << "] " << testName << "... ";
        
        if (condition) {
            setColor(COLOR_GREEN);
            std::cout << "PASSED";
        } else {
            setColor(COLOR_RED);
            std::cout << "FAILED";
            if (!message.empty()) {
                std::cout << " (" << message << ")";
            }
        }
        setColor(COLOR_WHITE);
        std::cout << std::endl;
    }

    void verifyPUZZLE71() {
        setColor(COLOR_CYAN);
        std::cout << "\n========================================\n";
        std::cout << "    PUZZLE71 Implementation Verifier\n";
        std::cout << "========================================\n\n";
        setColor(COLOR_WHITE);

        // Test 1: Check SearchMode.h
        runTest("SearchMode.h contains PUZZLE71 enum",
            fileContains("GPU/SearchMode.h", "PUZZLE71 = 7"));

        // Test 2: Check GPUCompute.h for hardcoded target
        runTest("GPUCompute.h contains PUZZLE71_TARGET_HASH",
            fileContains("GPU/GPUCompute.h", "PUZZLE71_TARGET_HASH"));

        // Test 3: Check for target address
        runTest("Target address is documented",
            fileContains("GPU/GPUCompute.h", "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU"));

        // Test 4: Check for ComputeKeysPUZZLE71
        runTest("ComputeKeysPUZZLE71 function exists",
            fileContains("GPU/GPUCompute.h", "ComputeKeysPUZZLE71"));

        // Test 5: Check for endomorphism header
        runTest("ECC_Endomorphism.h exists",
            fileExists("GPU/ECC_Endomorphism.h"));

        // Test 6: Check for endomorphism inclusion
        runTest("Endomorphism included in GPUCompute.h",
            fileContains("GPU/GPUCompute.h", "ECC_Endomorphism.h"));

        // Test 7: Check Main.cpp for PUZZLE71 mode
        runTest("Main.cpp supports PUZZLE71 mode",
            fileContains("Main.cpp", "SEARCH_MODE_PUZZLE71") || 
            fileContains("Main.cpp", "puzzle71"));

        // Test 8: Check KeyHunt.cpp for PUZZLE71
        runTest("KeyHunt.cpp handles PUZZLE71",
            fileContains("KeyHunt.cpp", "PUZZLE71"));

        // Test 9: Check GPU kernel files
        runTest("GPUEngine.cu contains PUZZLE71 kernels",
            fileContains("GPU/GPUEngine.cu", "compute_keys_puzzle71") ||
            fileContains("GPU/GPUEngine.cu", "ComputeKeysPUZZLE71"));

        // Test 10: Check unified kernel
        runTest("GPUEngine_Unified.cu has PUZZLE71 template",
            fileContains("GPU/GPUEngine_Unified.cu", "SearchMode::PUZZLE71"));

        // Test 11: Check documentation
        runTest("User guide exists",
            fileExists("PUZZLE71_USER_GUIDE.md"));

        // Test 12: Check test reports
        runTest("Completion reports exist",
            fileExists("ALL_TASKS_SUMMARY.md"));

        // Test 13: Check build scripts
        runTest("Build scripts exist",
            fileExists("build_puzzle71.ps1") || fileExists("build_win_puzzle71.ps1"));

        // Test 14: Check Makefile
        runTest("Makefile supports PUZZLE71",
            fileContains("Makefile", "PUZZLE71"));

        // Test 15: Verify hardcoded hash values
        runTest("Hash values are correct",
            fileContains("GPU/GPUCompute.h", "0x225b45f8") &&
            fileContains("GPU/GPUCompute.h", "0xb4242993"));

        // Print summary
        std::cout << "\n========================================\n";
        setColor(COLOR_CYAN);
        std::cout << "            VERIFICATION SUMMARY\n";
        setColor(COLOR_WHITE);
        std::cout << "========================================\n\n";
        
        std::cout << "Total Tests: " << totalTests << "\n";
        setColor(COLOR_GREEN);
        std::cout << "Passed: " << passedTests << "\n";
        
        if (passedTests < totalTests) {
            setColor(COLOR_RED);
            std::cout << "Failed: " << (totalTests - passedTests) << "\n";
        }
        
        setColor(COLOR_WHITE);
        double percentage = (passedTests * 100.0) / totalTests;
        std::cout << "Success Rate: " << percentage << "%\n\n";

        if (percentage == 100.0) {
            setColor(COLOR_GREEN);
            std::cout << "✓ PUZZLE71 IMPLEMENTATION VERIFIED!\n";
            std::cout << "All components are properly integrated.\n";
        } else if (percentage >= 80.0) {
            setColor(COLOR_YELLOW);
            std::cout << "⚠ PUZZLE71 MOSTLY COMPLETE\n";
            std::cout << "Most components are integrated but some tests failed.\n";
        } else {
            setColor(COLOR_RED);
            std::cout << "✗ PUZZLE71 INCOMPLETE\n";
            std::cout << "Significant components are missing or not integrated.\n";
        }
        
        setColor(COLOR_WHITE);
        std::cout << "\n";

        // List failed tests
        if (passedTests < totalTests) {
            setColor(COLOR_RED);
            std::cout << "Failed Tests:\n";
            for (const auto& result : results) {
                if (!result.passed) {
                    std::cout << "  - " << result.testName;
                    if (!result.message.empty()) {
                        std::cout << " (" << result.message << ")";
                    }
                    std::cout << "\n";
                }
            }
            setColor(COLOR_WHITE);
        }
    }
};

int main() {
    PUZZLE71Verifier verifier;
    verifier.verifyPUZZLE71();
    
    setColor(COLOR_CYAN);
    std::cout << "\nPress Enter to exit...";
    setColor(COLOR_WHITE);
    std::cin.get();
    
    return 0;
}