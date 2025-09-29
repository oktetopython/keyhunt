#include <stdio.h>
#include <string.h>
#include "Int.h"
#include "Point.h" 
#include "SECP256k1.h"

// Minimal dependencies test for our EC fixes
int main() {
    printf("=== P0 椭圆曲线修复验证测试 ===\n\n");
    
    try {
        // Create secp256k1 instance
        Secp256K1 secp;
        printf("1. 创建Secp256K1实例...\n");
        
        // Enable fast init to use our fixed generator table
        secp.SetFastInit(true);
        printf("2. 启用快速初始化模式...\n");
        
        // Initialize - this will call our fixed InitializeFastGeneratorTable
        printf("3. 初始化生成元表（测试InitializeFastGeneratorTable修复）...\n");
        secp.Init();
        printf("   ✓ 初始化完成\n");
        
        // Test 1: Get generator point
        Point G = secp.G;
        printf("\n4. 验证生成元点G:\n");
        printf("   G.x: %s\n", G.x.GetBase16().c_str());
        printf("   G.y: %s\n", G.y.GetBase16().c_str());
        
        // Known correct secp256k1 generator coordinates
        const char* expected_gx = "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798";
        const char* expected_gy = "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8";
        
        bool gx_correct = (G.x.GetBase16() == expected_gx);
        bool gy_correct = (G.y.GetBase16() == expected_gy);
        
        if (gx_correct && gy_correct) {
            printf("   ✓ 生成元点G坐标正确\n");
        } else {
            printf("   ✗ 生成元点G坐标错误\n");
            printf("   期望 G.x: %s\n", expected_gx);
            printf("   期望 G.y: %s\n", expected_gy);
            return 1;
        }
        
        // Test 2: ComputePublicKey with k=1 should return G
        printf("\n5. 测试ComputePublicKey修复 - k=1应返回G:\n");
        Int k1;
        k1.SetInt32(1);
        Point pub1 = secp.ComputePublicKey(&k1);
        
        printf("   ComputePublicKey(1):\n");
        printf("   pub1.x: %s\n", pub1.x.GetBase16().c_str());
        printf("   pub1.y: %s\n", pub1.y.GetBase16().c_str());
        
        bool pub1_x_correct = pub1.x.IsEqual(&G.x);
        bool pub1_y_correct = pub1.y.IsEqual(&G.y);
        
        if (pub1_x_correct && pub1_y_correct) {
            printf("   ✓ ComputePublicKey(1) = G (正确)\n");
        } else {
            printf("   ✗ ComputePublicKey(1) != G (错误)\n");
            printf("   x相等: %s, y相等: %s\n", 
                   pub1_x_correct ? "是" : "否", 
                   pub1_y_correct ? "是" : "否");
            return 1;
        }
        
        // Test 3: ComputePublicKey with k=2 should return 2*G
        printf("\n6. 测试ComputePublicKey修复 - k=2应返回2*G:\n");
        Int k2;
        k2.SetInt32(2);
        Point pub2 = secp.ComputePublicKey(&k2);
        
        printf("   ComputePublicKey(2):\n");
        printf("   pub2.x: %s\n", pub2.x.GetBase16().c_str());
        printf("   pub2.y: %s\n", pub2.y.GetBase16().c_str());
        
        // Compare with DoubleDirect(G)
        Point double_g = secp.DoubleDirect(G);
        printf("   DoubleDirect(G):\n");
        printf("   double_g.x: %s\n", double_g.x.GetBase16().c_str());
        printf("   double_g.y: %s\n", double_g.y.GetBase16().c_str());
        
        bool pub2_x_correct = pub2.x.IsEqual(&double_g.x);
        bool pub2_y_correct = pub2.y.IsEqual(&double_g.y);
        
        if (pub2_x_correct && pub2_y_correct) {
            printf("   ✓ ComputePublicKey(2) = DoubleDirect(G) (正确)\n");
        } else {
            printf("   ✗ ComputePublicKey(2) != DoubleDirect(G) (错误)\n");
            printf("   x相等: %s, y相等: %s\n", 
                   pub2_x_correct ? "是" : "否", 
                   pub2_y_correct ? "是" : "否");
            return 1;
        }
        
        // Test 4: Test with larger private key
        printf("\n7. 测试更大的私钥 k=1000:\n");
        Int k_large;
        k_large.SetInt32(1000);
        Point pub_large = secp.ComputePublicKey(&k_large);
        
        printf("   ComputePublicKey(1000):\n");
        printf("   pub_large.x: %s\n", pub_large.x.GetBase16().c_str());
        printf("   pub_large.y: %s\n", pub_large.y.GetBase16().c_str());
        
        // Verify it's not zero point and not same as G or 2*G
        bool not_zero = !pub_large.isZero();
        bool not_g = !(pub_large.x.IsEqual(&G.x) && pub_large.y.IsEqual(&G.y));
        bool not_2g = !(pub_large.x.IsEqual(&double_g.x) && pub_large.y.IsEqual(&double_g.y));
        
        if (not_zero && not_g && not_2g) {
            printf("   ✓ 大私钥计算结果合理（非零点，与G、2*G不同）\n");
        } else {
            printf("   ✗ 大私钥计算结果异常\n");
            printf("   非零点: %s, 不等于G: %s, 不等于2*G: %s\n",
                   not_zero ? "是" : "否",
                   not_g ? "是" : "否", 
                   not_2g ? "是" : "否");
            return 1;
        }
        
        printf("\n=== 所有测试通过! P0修复验证成功 ===\n");
        printf("ComputePublicKey和InitializeFastGeneratorTable修复工作正常\n");
        
        return 0;
        
    } catch (...) {
        printf("错误: 测试过程中发生异常\n");
        return 1;
    }
}