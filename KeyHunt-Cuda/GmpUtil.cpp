#include "GmpUtil.h"
#include "Int.h"
#include <string>
#include <cmath>

// Calculate percentage: (searched_count * 100) / total_range
double CalcPercantage(Int searchedCount, Int start, Int range)
{
	// Calculate: (searchedCount * 100) / range
	// searchedCount is the number of keys searched, not absolute position
	
	// Convert Int to double for percentage calculation
	double searched = searchedCount.ToDouble();
	double total = range.ToDouble();
	
	// Avoid division by zero
	if (total == 0.0) {
		return 0.0;
	}
	
	// Calculate percentage
	return (searched * 100.0) / total;
}
