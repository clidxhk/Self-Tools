import numpy as np
import re

def parse_soc_matrix(text):
    lines = text.strip().split('\n')
    soc_data = []
    
    for line in lines:
        # 匹配数据行
        match = re.match(r'^\s*(\d+)\s+(\d+)\s+\(\s*([-.0-9]+)\s*,\s*([-.0-9]+)\)\s+\(\s*([-.0-9]+)\s*,\s*([-.0-9]+)\)\s+\(\s*([-.0-9]+)\s*,\s*([-.0-9]+)\)', line)
        if match:
            t, s = int(match.group(1)), int(match.group(2))
            ms0 = complex(float(match.group(3)), float(match.group(4)))
            msm1 = complex(float(match.group(5)), float(match.group(6)))
            msp1 = complex(float(match.group(7)), float(match.group(8)))
            soc_data.append((t, s, ms0, msm1, msp1))
    return soc_data


def parse_s1_soc_matrix(text):
    lines = text.strip().split('\n')
    soc_data = []
    
    for line in lines:
        match = re.match(r'^\s*(\d+)\s+(\d+)\s+\(\s*([-.0-9]+)\s*,\s*([-.0-9]+)\)\s+\(\s*([-.0-9]+)\s*,\s*([-.0-9]+)\)\s+\(\s*([-.0-9]+)\s*,\s*([-.0-9]+)\)', line)
        if match:
            t, s = int(match.group(1)), int(match.group(2))
            # 只保留S1的数据
            if s == 1:
                ms0 = complex(float(match.group(3)), float(match.group(4)))
                msm1 = complex(float(match.group(5)), float(match.group(6)))
                msp1 = complex(float(match.group(7)), float(match.group(8)))
                soc_data.append((t, s, ms0, msm1, msp1))
    return soc_data


def calculate_soc(ms0, msm1, msp1):
    return np.sqrt(abs(ms0)**2 + abs(msm1)**2 + abs(msp1)**2)

def main():
    # 从文件读取数据
    with open('D:/Gaussian/SUSTech/2025/mTPA-RH.log', 'r') as f:
        input_data = f.read()
    
    soc_data = parse_s1_soc_matrix(input_data)
    
    # 按T态分组输出
    print("\n旋轨耦合常数计算结果:")
    print("-" * 40)
    print("三重态(T) 单重态(S)  SOC (cm⁻¹)")
    print("-" * 40)
    
    current_t = None
    for t, s, ms0, msm1, msp1 in soc_data:
        if current_t != t:
            if current_t is not None:
                print("-" * 40)
            current_t = t
        soc = calculate_soc(ms0, msm1, msp1)
        print(f"   T{t}        S{s}      {soc:8.2f}")

if __name__ == "__main__":
    main()