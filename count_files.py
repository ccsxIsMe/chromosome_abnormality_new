import os
from tabulate import tabulate  # 如果没有请 pip install tabulate，或者用下面的普通打印版

def count_files_in_subdirs(base_path):
    if not os.path.exists(base_path):
        print(f"错误: 路径 {base_path} 不存在")
        return

    stats = []
    total_count = 0
    
    # 获取所有子项并排序
    subdirs = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

    print(f"正在扫描路径: {base_path} ...\n")

    for subdir in subdirs:
        subdir_path = os.path.join(base_path, subdir)
        # 统计该子文件夹下的文件数量（排除子文件夹，只算文件）
        file_count = len([f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))])
        
        stats.append([subdir, file_count])
        total_count += file_count

    # 打印结果
    headers = ["子文件夹名称 (Subdirectory)", "文件数量 (File Count)"]
    print(tabulate(stats, headers=headers, tablefmt="grid"))
    print(f"\n汇总: 共找到 {len(subdirs)} 个子文件夹，总计 {total_count} 个文件。")

if __name__ == "__main__":
    target_path = "/data5/chensx/MyProject/UAE/data/arrangement_merged"
    count_files_in_subdirs(target_path)