#include "GmpUtil.h"
#include <gmp.h>
#include <gmpxx.h>

// Calculate percentage: (searched_count * 100) / total_range
double CalcPercantage(Int searchedCount, Int start, Int range)
{
	// Calculate: (searchedCount * 100) / range
	// searchedCount is the number of keys searched, not absolute position
	mpz_class x(searchedCount.GetBase16().c_str(), 16);
	mpz_class r(range.GetBase16().c_str(), 16);

	// x = searchedCount * 100
	x = x * 100;

	// Calculate percentage
	mpf_class y(x);
	y = y / mpf_class(r);
	return y.get_d();
}
