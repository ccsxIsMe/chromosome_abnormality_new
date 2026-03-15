import os

def delete_dup_files(root_folder):
    """
    递归删除指定文件夹及其子文件夹中，文件名包含 _dup 的所有文件
    :param root_folder: 要扫描的根文件夹路径
    """
    # 统计删除数量
    deleted_count = 0
    skipped_count = 0

    print(f"开始扫描文件夹：{root_folder}\n")

    # 递归遍历所有文件和子文件夹
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            # 判断文件名是否包含 _dup
            if "_dup" in filename:
                file_path = os.path.join(dirpath, filename)
                
                try:
                    # 删除文件
                    os.remove(file_path)
                    print(f"已删除：{file_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"删除失败：{file_path}，错误：{str(e)}")
                    skipped_count += 1

    # 输出结果
    print("\n" + "-" * 50)
    print(f"任务完成！")
    print(f"成功删除文件：{deleted_count} 个")
    print(f"删除失败文件：{skipped_count} 个")

# ====================== 在这里修改你的目标文件夹路径 ======================
TARGET_FOLDER = r"/data5/chensx/MyProject/UAE/data/splits-case"
# ========================================================================

if __name__ == "__main__":
    # 安全确认
    confirm = input(f"⚠️  确定要删除【{TARGET_FOLDER}】下所有包含 _dup 的文件吗？(y/n)：")
    if confirm.lower() == "y":
        delete_dup_files(TARGET_FOLDER)
    else:
        print("已取消操作")