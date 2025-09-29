#include "Timer.h"
#include "KeyHunt.h"
#include "Base58.h"
#include "CmdParse.h"
#include "Constants.h"
#include <fstream>
#include <string>
#include <string.h>
#include <stdexcept>
#include <cassert>
#include <algorithm>
#include <inttypes.h>
#ifndef WIN64
#include <signal.h>
#include <unistd.h>
#endif

#define RELEASE KEYHUNT_VERSION_STRING

using namespace std;
bool should_exit = false;

// ----------------------------------------------------------------------------
void usage()
{
	printf("KeyHunt-Cuda [OPTIONS...] [TARGETS]\n");
	printf("Where TARGETS is one address/xpont, or multiple hashes/xpoints file\n\n");

	printf("-h, --help                               : Display this message\n");
	printf("-c, --check                              : Check the working of the codes\n");
    printf("-u, --uncomp                             : Force uncompressed search mode\n");
    printf("-b, --both                               : Search both compressed and uncompressed points\n");
    printf("-g, --gpu                                : (Optional) explicit GPU enable flag\n");
    printf("--gpui GPU ids: 0,1,...                  : List of GPU(s) to use, default is 0\n");
    printf("--gpux GPU gridsize: g0x,g0y,g1x,g1y,... : Specify GPU(s) kernel gridsize, default is 8*(Device MP count),128\n");
	printf("-i, --in FILE                            : Read rmd160 hashes or xpoints from FILE, should be in binary format with sorted\n");
	printf("-o, --out FILE                           : Write keys to FILE, default: Found.txt\n");
	printf("-m, --mode MODE                          : Specify search mode where MODE is\n");
	printf("                                               ADDRESS  : for single address\n");
	printf("                                               ADDRESSES: for multiple hashes/addresses\n");
	printf("                                               XPOINT   : for single xpoint\n");
	printf("                                               XPOINTS  : for multiple xpoints\n");
	printf("                                               PUZZLE71 : Specialized mode for Bitcoin Puzzle #71\n");
	printf("--coin BTC/ETH                           : Specify Coin name to search\n");
	printf("                                               BTC: available mode :-\n");
	printf("                                                   ADDRESS, ADDRESSES, XPOINT, XPOINTS\n");
	printf("                                               ETH: available mode :-\n");
	printf("                                                   ADDRESS, ADDRESSES\n");
	printf("-l, --list                               : List cuda enabled devices\n");
	printf("--range KEYSPACE                         : Specify the range:\n");
	printf("                                               START:END\n");
	printf("                                               START:+COUNT\n");
	printf("                                               START\n");
	printf("                                               :END\n");
	printf("                                               :+COUNT\n");
	printf("                                               Where START, END, COUNT are in hex format\n");
	printf("-r, --rkey Rkey                          : Random key interval in MegaKeys, default is disabled\n");
	printf("-v, --version                            : Show version\n");
}

// ----------------------------------------------------------------------------

void getInts(string name, vector<int>& tokens, const string& text, char sep)
{

	size_t start = 0, end = 0;
	tokens.clear();
	int item;

	try {

		while ((end = text.find(sep, start)) != string::npos) {
			item = std::stoi(text.substr(start, end - start));
			tokens.push_back(item);
			start = end + 1;
		}

		item = std::stoi(text.substr(start));
		tokens.push_back(item);

	}
	catch (std::invalid_argument&) {

		printf("Invalid %s argument, number expected\n", name.c_str());
		usage();
		exit(-1);

	}

}

// ----------------------------------------------------------------------------

int parseSearchMode(const std::string& s)
{
	std::string stype = s;
	std::transform(stype.begin(), stype.end(), stype.begin(), ::tolower);

	if (stype == "address") {
		return SEARCH_MODE_SA;
	}

	if (stype == "xpoint") {
		return SEARCH_MODE_SX;
	}

	if (stype == "addresses") {
		return SEARCH_MODE_MA;
	}

	if (stype == "xpoints") {
		return SEARCH_MODE_MX;
	}

	if (stype == "puzzle71") {
		return SEARCH_MODE_PUZZLE71;
	}

	printf("Invalid search mode format: %s", stype.c_str());
	usage();
	exit(-1);
}

// ----------------------------------------------------------------------------

int parseCoinType(const std::string& s)
{
	std::string stype = s;
	std::transform(stype.begin(), stype.end(), stype.begin(), ::tolower);

	if (stype == "btc") {
		return COIN_BTC;
	}

	if (stype == "eth") {
		return COIN_ETH;
	}

	printf("Invalid coin name: %s", stype.c_str());
	usage();
	exit(-1);
}

// ----------------------------------------------------------------------------

bool parseRange(const std::string& s, Int& start, Int& end)
{
	size_t pos = s.find(':');

	if (pos == std::string::npos) {
		start.SetBase16(s.c_str());
		end.Set(&start);
		end.Add(KeyHuntConstants::DEFAULT_RANGE_END);
	}
	else {
		std::string left = s.substr(0, pos);

		if (left.length() == 0) {
			start.SetInt32(1);
		}
		else {
			start.SetBase16(left.c_str());
		}

		std::string right = s.substr(pos + 1);

		if (right[0] == '+') {
			Int t;
			t.SetBase16(right.substr(1).c_str());
			end.Set(&start);
			end.Add(&t);
		}
		else {
			end.SetBase16(right.c_str());
		}
	}

	return true;
}

#ifdef WIN64
BOOL WINAPI CtrlHandler(DWORD fdwCtrlType)
{
	switch (fdwCtrlType) {
	case CTRL_C_EVENT:
		//printf("\n\nCtrl-C event\n\n");
		should_exit = true;
		return TRUE;

	default:
		return TRUE;
	}
}
#else
void CtrlHandler(int signum) {
	printf("\n\nBYE\n");
	exit(signum);
}
#endif

// ----------------------------------------------------------------------------
// 应用程序类，用于重构长main函数
class Application {
private:
	// 配置参数
	bool gpuEnable;
	bool gpuAutoGrid;
	int compMode;
	vector<int> gpuId;
	vector<int> gridSize;
	string outputFile;
	string inputFile;
	string address;
	string xpoint;
	std::vector<unsigned char> hashORxpoint;
	uint32_t maxFound;
	uint64_t rKey;
	Int rangeStart;
	Int rangeEnd;
	int searchMode;
	int coinType;

public:
	Application() : 
		gpuEnable(true), gpuAutoGrid(true), compMode(SEARCH_COMPRESSED), 
		gpuId({0}), outputFile("Found.txt"), maxFound(KeyHuntConstants::DEFAULT_MAX_FOUND),
		rKey(0), searchMode(0), coinType(COIN_BTC) {
		rangeStart.SetInt32(0);
		rangeEnd.SetInt32(0);
		hashORxpoint.clear();
	}

	// 解析命令行参数
	bool parseArguments(int argc, char** argv, CmdParse& parser) {
		parser.add("-h", "--help", false);
		parser.add("-c", "--check", false);
		parser.add("-l", "--list", false);
		parser.add("-u", "--uncomp", false);
		parser.add("-b", "--both", false);
		parser.add("-g", "--gpu", false);
		parser.add("", "--gpui", true);
		parser.add("", "--gpux", true);
		parser.add("-i", "--in", true);
		parser.add("-o", "--out", true);
		parser.add("-m", "--mode", true);
		parser.add("", "--coin", true);
		parser.add("", "--range", true);
		parser.add("-r", "--rkey", true);
		parser.add("-v", "--version", false);

		if (argc == 1) {
			usage();
			return false;
		}
		
		try {
			parser.parse(argc, argv);
		}
		catch (std::string err) {
			printf("Error: %s\n", err.c_str());
			usage();
			exit(-1);
		}
		
		return true;
	}

	// 处理命令行选项
	int processOptions(const std::vector<OptArg>& args) {
		for (unsigned int i = 0; i < args.size(); i++) {
			OptArg optArg = args[i];
			std::string opt = args[i].option;

			try {
				if (optArg.equals("-h", "--help")) {
					usage();
					return 0;
				}
				else if (optArg.equals("-c", "--check")) {
					printf("KeyHunt-Cuda v%s\n\n", RELEASE);
					printf("\nChecking... Secp256K1\n\n");
					Secp256K1* secp = new Secp256K1();
					secp->Init();
					secp->Check();
					printf("\n\nChecking... Int\n\n");
					Int* K = new Int();
					K->SetBase16("3EF7CEF65557B61DC4FF2313D0049C584017659A32B002C105D04A19DA52CB47");
					K->Check();
					delete secp;
					delete K;
					printf("\n\nChecked successfully\n\n");
					return 0;
				}
				else if (optArg.equals("-l", "--list")) {
#ifdef WITHGPU
					GPUEngine::PrintCudaInfo();
#else
					printf("GPU code not compiled, use -DWITHGPU when compiling.\n");
#endif
					return 0;
				}
				else if (optArg.equals("-u", "--uncomp")) {
					compMode = SEARCH_UNCOMPRESSED;
				}
				else if (optArg.equals("-b", "--both")) {
					compMode = SEARCH_BOTH;
				}
			else if (optArg.equals("-g", "--gpu")) {
				gpuEnable = true;
			}
				else if (optArg.equals("", "--gpui")) {
					string ids = optArg.arg;
					getInts("--gpui", gpuId, ids, ',');
				}
				else if (optArg.equals("", "--gpux")) {
					string grids = optArg.arg;
					getInts("--gpux", gridSize, grids, ',');
					gpuAutoGrid = false;
				}
				else if (optArg.equals("-i", "--in")) {
					inputFile = optArg.arg;
				}
				else if (optArg.equals("-o", "--out")) {
					outputFile = optArg.arg;
				}
				else if (optArg.equals("-m", "--mode")) {
					searchMode = parseSearchMode(optArg.arg);
				}
				else if (optArg.equals("", "--coin")) {
					coinType = parseCoinType(optArg.arg);
				}
				else if (optArg.equals("", "--range")) {
					std::string range = optArg.arg;
					parseRange(range, rangeStart, rangeEnd);
				}
				else if (optArg.equals("-r", "--rkey")) {
					rKey = std::stoull(optArg.arg);
				}
				else if (optArg.equals("-v", "--version")) {
					printf("KeyHunt-Cuda %s (%s)\n", KEYHUNT_VERSION, KEYHUNT_BUILD_DATE);
					printf("Build with unified kernel interface and LDG cache optimization\n");
					return 0;
				}
			}
			catch (std::string err) {
				printf("Error: %s\n", err.c_str());
				usage();
				return -1;
			}
		}
		
		return -2; // 继续处理
	}

	// 验证配置
	bool validateConfig() {
		if (coinType == COIN_ETH && (searchMode == SEARCH_MODE_SX || searchMode == SEARCH_MODE_MX/* || compMode == SEARCH_COMPRESSED*/)) {
			printf("Error: %s\n", "Wrong search or compress mode provided for ETH coin type");
			usage();
			return false;
		}
		
		if (coinType == COIN_ETH) {
			compMode = SEARCH_UNCOMPRESSED;
		}

		return true;
	}

	// 解析操作数
	int parseOperands(const std::vector<std::string>& ops) {
		if (ops.size() == 0) {
			// read from file
			// PUZZLE71 mode doesn't need input file or operands
			if (inputFile.size() == 0 && searchMode != SEARCH_MODE_PUZZLE71) {
				printf("Error: %s\n", "Missing arguments");
				usage();
				return -1;
			}
			if (searchMode != SEARCH_MODE_MA && searchMode != SEARCH_MODE_MX && searchMode != SEARCH_MODE_PUZZLE71) {
				printf("Error: %s\n", "Wrong search mode provided for multiple addresses or xpoints");
				usage();
				return -1;
			}
		}
		else {
			// read from cmdline
			// PUZZLE71 mode doesn't need an address argument
			if (searchMode == SEARCH_MODE_PUZZLE71 ? ops.size() > 1 : ops.size() != 1) {
				printf("Error: %s\n", "Wrong args or more than one address or xpoint are provided, use inputFile for multiple addresses or xpoints");
				usage();
				return -1;
			}
			if (searchMode != SEARCH_MODE_SA && searchMode != SEARCH_MODE_SX && searchMode != SEARCH_MODE_PUZZLE71) {
				printf("Error: %s\n", "Wrong search mode provided for single address or xpoint");
				usage();
				return -1;
			}

			switch (searchMode) {
			case (int)SEARCH_MODE_SA:
			{
				address = ops[0];
				if (coinType == COIN_BTC) {
					if (address.length() < 30 || address[0] != '1') {
						printf("Error: %s\n", "Invalid address, must have Bitcoin P2PKH address or Ethereum address");
						usage();
						return -1;
					}
					else {
						if (DecodeBase58(address, hashORxpoint)) {
							hashORxpoint.erase(hashORxpoint.begin() + 0);
							hashORxpoint.erase(hashORxpoint.begin() + 20, hashORxpoint.begin() + 24);
							assert(hashORxpoint.size() == 20);
						}
					}
				}
				else {
					if (address.length() != 42 || address[0] != '0' || address[1] != 'x') {
						printf("Error: %s\n", "Invalid Ethereum address");
						usage();
						return -1;
					}
					address.erase(0, 2);
					for (int i = 0; i < 40; i += 2) {
						uint8_t c = 0;
						for (size_t j = 0; j < 2; j++) {
							uint32_t c0 = (uint32_t)address[i + j];
							uint8_t c2 = (uint8_t)strtol((char*)&c0, NULL, 16);
							if (j == 0)
								c2 = c2 << 4;
							c |= c2;
						}
						hashORxpoint.push_back(c);
					}
					assert(hashORxpoint.size() == 20);
				}
			}
			break;
			case (int)SEARCH_MODE_SX:
			{
				unsigned char xpbytes[32];
				xpoint = ops[0];
				Int* xp = new Int();
				xp->SetBase16(xpoint.c_str());
				xp->Get32Bytes(xpbytes);
				for (int i = 0; i < 32; i++)
					hashORxpoint.push_back(xpbytes[i]);
				delete xp;
				if (hashORxpoint.size() != 32) {
					printf("Error: %s\n", "Invalid xpoint");
					usage();
					return -1;
				}
			}
			break;
		case (int)SEARCH_MODE_PUZZLE71:
			{
				// For PUZZLE71 mode, use hardcoded Bitcoin Puzzle #71 address
				// Address: 1HBtApAFA9B2YZw3G2YKSMCtb3dVnjuNe2
				// The actual target hash will be handled in the GPU kernel
				if (ops.size() == 1) {
					// If user provided an address, we can accept it for compatibility
					address = ops[0];
				} else {
					// No address provided, use the hardcoded Puzzle #71 address
					address = "1HBtApAFA9B2YZw3G2YKSMCtb3dVnjuNe2";
				}
				
				// Validate the address format
				if (address.length() < 30 || address[0] != '1') {
					printf("Error: %s\n", "Invalid address for PUZZLE71 mode");
					usage();
					return -1;
				}
				// Decode the provided address (will be replaced with hardcoded target in kernel)
				if (DecodeBase58(address, hashORxpoint)) {
					hashORxpoint.erase(hashORxpoint.begin() + 0);
					hashORxpoint.erase(hashORxpoint.begin() + 20, hashORxpoint.begin() + 24);
					assert(hashORxpoint.size() == 20);
				}
				
				// Auto-set range for PUZZLE71 if not already set
				if (rangeStart.GetBitLength() <= 0) {
					// Puzzle #71 range: [2^70, 2^71)
					rangeStart.SetBase16("40000000000000000");  // 2^70
					rangeEnd.SetBase16("80000000000000000");    // 2^71 - 1
					printf("Auto-set PUZZLE71 range: [2^70, 2^71)\n");
				}
			}
			break;
			default:
				printf("Error: %s\n", "Invalid search mode for single address or xpoint");
				usage();
				return -1;
				break;
			}
		}
		
		return 0;
	}

	// 配置GPU参数
	bool configureGPU() {
		printf("[configureGPU] gpuId.size()=%zu, gridSize.size()=%zu\n", gpuId.size(), gridSize.size());
		if (gridSize.size() == 0) {
			for (int i = 0; i < gpuId.size(); i++) {
				gridSize.push_back(-1);
				gridSize.push_back(KeyHuntConstants::DEFAULT_GPU_THREADS_PER_BLOCK);
			}
			printf("[configureGPU] After init: gridSize.size()=%zu\n", gridSize.size());
		}
		
		if (gridSize.size() != gpuId.size() * 2) {
			printf("Error: %s\n", "Invalid gridSize or gpuId argument, must have coherent size\n");
			usage();
			return false;
		}

		return true;
	}

	// 验证范围
	bool validateRange() {
		if (rangeStart.GetBitLength() <= 0) {
			printf("Error: Invalid start range, provide start range at least, end range would be: start range + 0x%" PRIx64 "\n", KeyHuntConstants::DEFAULT_RANGE_END);
			usage();
			return false;
		}
		return true;
	}

	void printConfig() {
		printf("\n");
		printf("KeyHunt-Cuda %s (%s)\n", KEYHUNT_VERSION, KEYHUNT_BUILD_DATE);
		printf("Unified kernel interface with LDG cache optimization\n");
		printf("\n");
		if (coinType == COIN_BTC)
			printf("COMP MODE    : %s\n", compMode == SEARCH_COMPRESSED ? "COMPRESSED" : (compMode == SEARCH_UNCOMPRESSED ? "UNCOMPRESSED" : "COMPRESSED & UNCOMPRESSED"));
		printf("COIN TYPE    : %s\n", coinType == COIN_BTC ? "BITCOIN" : "ETHEREUM");
		printf("SEARCH MODE  : %s\n", 
			searchMode == (int)SEARCH_MODE_MA ? "Multi Address" : 
			(searchMode == (int)SEARCH_MODE_SA ? "Single Address" : 
			(searchMode == (int)SEARCH_MODE_MX ? "Multi X Points" : 
			(searchMode == (int)SEARCH_MODE_SX ? "Single X Point" : 
			(searchMode == (int)SEARCH_MODE_PUZZLE71 ? "Puzzle #71" : "Unknown")))));
		printf("DEVICE       : GPU\n");
		if (gpuEnable) {
			printf("GPU IDS      : ");
			for (int i = 0; i < gpuId.size(); i++) {
				printf("%d", gpuId.at(i));
				if (i + 1 < gpuId.size())
					printf(", ");
			}
			printf("\n");
			printf("GPU GRIDSIZE : ");
			for (int i = 0; i < gridSize.size(); i++) {
				printf("%d", gridSize.at(i));
				if (i + 1 < gridSize.size()) {
					if ((i + 1) % 2 != 0) {
						printf("x");
					}
					else {
						printf(", ");
					}

				}
			}
			if (gpuAutoGrid)
				printf(" (Auto grid size)\n");
			else
				printf("\n");
		}
		printf("RKEY         : %" PRIu64 " Mkeys\n", rKey);
		printf("MAX FOUND    : %d\n", maxFound);
		if (coinType == COIN_BTC) {
			switch (searchMode) {
			case (int)SEARCH_MODE_MA:
				printf("BTC HASH160s : %s\n", inputFile.c_str());
				break;
			case (int)SEARCH_MODE_SA:
				printf("BTC ADDRESS  : %s\n", address.c_str());
				break;
			case (int)SEARCH_MODE_MX:
				printf("BTC XPOINTS  : %s\n", inputFile.c_str());
				break;
			case (int)SEARCH_MODE_SX:
				printf("BTC XPOINT   : %s\n", xpoint.c_str());
				break;
			case (int)SEARCH_MODE_PUZZLE71:
				printf("TARGET       : Bitcoin Puzzle #71\n");
				printf("TARGET ADDR  : 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU (hardcoded)\n");
				break;
			default:
				break;
			}
		}
		else {
			switch (searchMode) {
			case (int)SEARCH_MODE_MA:
				printf("ETH ADDRESSES: %s\n", inputFile.c_str());
				break;
			case (int)SEARCH_MODE_SA:
				printf("ETH ADDRESS  : 0x%s\n", address.c_str());
				break;
			default:
				break;
			}
		}
		printf("OUTPUT FILE  : %s\n", outputFile.c_str());
	}

	// 运行搜索
	int runSearch() {
		if (!gpuEnable) {
			printf("Error: GPU execution must be enabled in this build\n");
			return -1;
		}
#ifdef WIN64
		printf("Starting GPU search (Windows handler installed)\n");
#else
		printf("Starting GPU search\n");
#endif
#ifdef WIN64
		if (SetConsoleCtrlHandler(CtrlHandler, TRUE)) {
			KeyHunt* v;
			switch (searchMode) {
			case (int)SEARCH_MODE_MA:
			case (int)SEARCH_MODE_MX:
				v = new KeyHunt(inputFile, compMode, searchMode, coinType, gpuEnable, outputFile,
					maxFound, rKey, rangeStart.GetBase16(), rangeEnd.GetBase16(), should_exit);
				break;
			case (int)SEARCH_MODE_SA:
			case (int)SEARCH_MODE_SX:
			case (int)SEARCH_MODE_PUZZLE71:
				v = new KeyHunt(hashORxpoint, compMode, searchMode, coinType, gpuEnable, outputFile,
					maxFound, rKey, rangeStart.GetBase16(), rangeEnd.GetBase16(), should_exit);
				break;
			default:
				printf("\n\nNothing to do, exiting\n");
				return 0;
				break;
			}
			v->Search(gpuId, gridSize, should_exit);
			delete v;
			printf("\n\nBYE\n");
			return 0;
		}
		else {
			printf("Error: could not set control-c handler\n");
			return -1;
		}
#else
		signal(SIGINT, CtrlHandler);
		KeyHunt* v;
		switch (searchMode) {
		case (int)SEARCH_MODE_MA:
		case (int)SEARCH_MODE_MX:
		v = new KeyHunt(inputFile, compMode, searchMode, coinType, gpuEnable, outputFile,
			maxFound, rKey, rangeStart.GetBase16(), rangeEnd.GetBase16(), should_exit);
			break;
	case (int)SEARCH_MODE_SA:
	case (int)SEARCH_MODE_SX:
	case (int)SEARCH_MODE_PUZZLE71:
	v = new KeyHunt(hashORxpoint, compMode, searchMode, coinType, gpuEnable, outputFile,
		maxFound, rKey, rangeStart.GetBase16(), rangeEnd.GetBase16(), should_exit);
		break;
		default:
			printf("\n\nNothing to do, exiting\n");
			return 0;
			break;
		}
	v->Search(gpuId, gridSize, should_exit);
		delete v;
		return 0;
#endif
	}
};

int main(int argc, char** argv)
{
	// Global Init
	Timer::Init();
	rseed(Timer::getSeed32());

	// 创建应用程序实例
	Application app;

	// cmd args parsing
	CmdParse parser;
	if (!app.parseArguments(argc, argv, parser)) {
		return 0;
	}
	
	std::vector<OptArg> args = parser.getArgs();
	int result = app.processOptions(args);
	if (result != -2) {
		return result;
	}

	// 验证配置
	if (!app.validateConfig()) {
		return -1;
	}

	// Parse operands
	std::vector<std::string> ops = parser.getOperands();
	for(size_t i = 0; i < ops.size(); i++) {
	}
	result = app.parseOperands(ops);
	if (result != 0) {
		return result;
	}

	// 配置GPU
	if (!app.configureGPU()) {
		return -1;
	}

	// 验证范围
	if (!app.validateRange()) {
		return -1;
	}

	// 打印配置信息
	app.printConfig();

	// 运行搜索
	return app.runSearch();
}