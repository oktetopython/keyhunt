#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#define SEARCH_MODE_PUZZLE71 7

int parseSearchMode(const std::string& s)
{
    std::string stype = s;
    std::transform(stype.begin(), stype.end(), stype.begin(), ::tolower);
    
    std::cout << "Input: '" << s << "', Lowercase: '" << stype << "'" << std::endl;
    
    if (stype == "address") {
        return 2;  // SEARCH_MODE_SA
    }
    
    if (stype == "xpoint") {
        return 4;  // SEARCH_MODE_SX
    }
    
    if (stype == "addresses") {
        return 1;  // SEARCH_MODE_MA
    }
    
    if (stype == "xpoints") {
        return 3;  // SEARCH_MODE_MX
    }
    
    if (stype == "puzzle71") {
        return SEARCH_MODE_PUZZLE71;
    }
    
    std::cout << "Invalid search mode format: " << stype << std::endl;
    return -1;
}

int main() {
    std::vector<std::string> ops = {};  // no operands
    int searchMode = parseSearchMode("puzzle71");
    
    std::cout << "searchMode = " << searchMode << std::endl;
    std::cout << "ops.size() = " << ops.size() << std::endl;
    
    // Test the logic from line 404
    bool condition = (searchMode == SEARCH_MODE_PUZZLE71 ? ops.size() > 1 : ops.size() != 1);
    std::cout << "Condition result: " << condition << std::endl;
    
    if (condition) {
        std::cout << "ERROR: Would fail the check" << std::endl;
    } else {
        std::cout << "PASS: Would pass the check" << std::endl;
    }
    
    return 0;
}