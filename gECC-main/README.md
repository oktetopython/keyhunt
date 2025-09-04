# lib gECC

## Introduction
This project presents gECC, a versatile framework for ECC optimized for GPU architectures, specifically engineered to achieve high-throughput performance in EC operations. To maximize throughput, gECC incorporates batch-based execution (using Montgomery’s trick) of EC operations and microarchitecture-level optimization of modular arithmetic. 

Copyright (C) 2024, [BDTS/STCS/CGCL](http://grid.hust.edu.cn/) and [Huazhong University of Science and Technology](https://www.hust.edu.cn/).

## 📄 Publication
Our work on **gECC** has been **accepted to appear** in *ACM Transactions on Architecture and Code Optimization (TACO)*. 

- Title: "gECC: A GPU-based high-throughput framework for Elliptic Curve Cryptography"
- Authors: Qian Xiong, Weiliang Ma, Xuanhua Shi, Yongluan Zhou, Hai Jin, Kaiyi Huang, Haozhou Wang, Zhengru Wang
- Journal: ACM Transactions on Architecture and Code Optimization (TACO)
- Status: Accepted, to appear
- Preprint: [gECC](https://arxiv.org/abs/2501.03245) available on arXiv

📢 We will update the final version and BibTeX entry once the paper is published online.


## Files
| Files | description | 
| -------- | -------- | 
| test | all performance analysis benchmarks |
| scripts | define finite field related parameters, generate test data for benchmark. | 
| gecc/arith | implemente ec operation (with multiple coordinate system) and modular operation on finite filed|
| gecc/ecdsa | implemente ECDSA algorithm|

## Prerequisites

### GTest

```
wget https://github.com/google/googletest/archive/release-1.10.0.tar.gz
tar xzvf release-1.10.0.tar.gz
cmake -BBuild -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${HOME}/.local/opt/gtest .
cmake --build Build --target install
export GTEST_ROOT=${HOME}/.local/opt/gtest
```

## Build and Run

To evaluate performance:
```
python3 ./dev-support/build.py -R -A 80
./bench.sh
```

## Support or Contact
gECC is developed in National Engineering Research Center for Big Data Technology and System, Cluster and Grid Computing Lab, Services Computing Technology and System Lab, School of Computer Science and Technology, Huazhong University of Science and Technology, Wuhan, China by Qian Xiong(qianxiong@hust.edu.cn), Weiliang Ma(weiliangma@hust.edu.cn) and Xuanhua Shi(xhshi@hust.edu.cn).

If you have any questions, please contact Qian Xiong(qianxiong@hust.edu.cn), Weiliang Ma(weiliangma@hust.edu.cn) and Xuanhua Shi(xhshi@hust.edu.cn). We welcome you to commit your modification to support our project.
