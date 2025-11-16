import os

base_dir = '.'

main_folders = ['train', 'test', 'val']
categories = ['NORMAL', 'PNEUMONIA']

for folder in main_folders:
    print(f"\n--- Đang xử lý thư mục: {folder} ---")

    # Lặp qua từng lớp (NORMAL, PNEUMONIA)
    for category in categories:
        dir_path = os.path.join(base_dir, folder, category)

        try:
            files = os.listdir(dir_path)

            file_count = len(files)
            print(f"  {category}: {file_count} files")
            
        except FileNotFoundError:
            print(f"  Lỗi: Không tìm thấy thư mục: {dir_path}")
        except Exception as e:
            print(f"  Lỗi khi đọc {dir_path}: {e}")